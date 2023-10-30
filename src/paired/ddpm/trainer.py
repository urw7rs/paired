import functools
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from flax import jax_utils, struct
from flax.training import common_utils, train_state
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training.dynamic_scale import DynamicScale
from jax import random, tree_util
from tqdm import tqdm

from . import sampler as ddpm_sampler
from .models import UNet


Array = jax.Array


class TrainState(train_state.TrainState):
    dynamic_scale: DynamicScale
    schedule: ddpm_sampler.Linear

@struct.dataclass
class HyperParams:
    batch_size: int = 128
    height: int = 32
    width: int = 32
    channels: int = 3
    timesteps: int = 1000

    seed: int = 42

    learning_rate: float = 2e-4
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    warmup_steps: int = 5000
    train_iterations: int = 800_000


def create_state(key, hparams: HyperParams):
    dropout_key, params_key = random.split(key)

    model = UNet(
        in_channels=3,
        pos_dim=128,
        emb_dim=512,
        drop_rate=0.1,
        channels_per_depth=(128, 256, 256, 256),
        num_blocks=2,
        attention_depths=(2,),
    )

    dummy_x = jnp.empty(
        (hparams.batch_size, hparams.height, hparams.width, hparams.channels)
    )
    dummy_t = jnp.empty((hparams.batch_size,))

    variables = model.init(params_key, dummy_x, dummy_t, training=False)
    state, params = variables.pop("params")
    del state

    learning_rate_schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=hparams.learning_rate / hparams.warmup_steps,
                end_value=hparams.learning_rate,
                transition_steps=hparams.warmup_steps,
            ),
        ],
        [hparams.warmup_steps],
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(hparams.grad_clip_norm),
            optax.adam(learning_rate=learning_rate_schedule),
            optax.ema(decay=hparams.ema_decay),
        ),
        dynamic_scale=DynamicScale(growth_factor=10, growth_interval=1),
        schedule=ddpm_sampler.Linear.create(hparams.timesteps),
    )

def preferred_dtype(use_mixed_precision):
    platform = jax.local_devices()[0].platform
    if use_mixed_precision:
        if platform == "tpu":
            return jnp.bfloat16
        elif platform == "gpu":
            return jnp.float16
        return jnp.float32



def simple_loss(params, state, dropout_key, alpha_bar_t, image, timestep, noise):
    x_t = ddpm_sampler.forward_process(alpha_bar_t, image, noise)

    noise_in_x_t = state.apply_fn(
        {"params": params},
        x_t,
        timestep,
        training=True,
        rngs={"dropout": dropout_key},
    )

    loss = optax.l2_loss(predictions=noise_in_x_t, targets=noise)
    return jnp.mean(loss)


def train_step(state: TrainState, image, key):
    logs = {"total_loss": jnp.ones(4, 4).mean(), "x_loss": jnp.ones(4, 4).mean(), "y_loss": jnp.ones(4, 4).mean()}
    return logs, state
    schedule = state.schedule

    key = random.fold_in(state.key, state.step)

    timestep_key, noise_key, dropout_key = random.split(key, num=3)

    leading_dims = image.shape[:-3]
    timestep = random.randint(
        timestep_key,
        shape=leading_dims,
        minval=1,
        maxval=schedule.timesteps,
    )

    noise = random.normal(noise_key, shape=image.shape)

    alpha_bar_t = schedule.alpha_bar[timestep]

    def loss_fn(params):
        return jnp.mean(
            simple_loss(params, state, dropout_key, alpha_bar_t, image, timestep, noise)
        )

    if state.dynamic_scale:
        loss_grad_fn = state.dynamic_scale.value_and_grad(loss_fn, axis_name="device")
        dynamic_scale, is_fin, loss, grads = loss_grad_fn(state.params)
        state = state.replace(dynamic_scale=dynamic_scale)

        new_state = state.apply_gradients(grads=grads)
    else:
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name="device")

    new_state = state.apply_gradients(grads=grads)
    loss = jax.lax.pmean(loss, axis_name="device")

    if state.dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        select_fn = functools.partial(jnp.where, is_fin)
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                select_fn, new_state.opt_state, state.opt_state
            ),
            params=jax.tree_util.tree_map(select_fn, new_state.params, state.params),
        )

    logs = {"losses": loss}
    return logs, new_state


def eval_step(state: TrainState, image, key):
    logs = {}
    return logs


def load_best_ckpt(ckpt: str):
    def best_fn(metrics: Mapping[str, float]) -> float:
        return metrics["FID_k"]

    best_ckpt = ckpt / "best"
    best_ckpt.mkdir(exist_ok=True)

    best_ckpt_mngr = ocp.CheckpointManager(
        best_ckpt,
        {
            "state": ocp.PyTreeCheckpointer(),
            "metrics": ocp.PyTreeCheckpointer(),
            },
        ocp.CheckpointManagerOptions(max_to_keep=5, best_fn=best_fn, best_mode="min"),
    )
    return best_ckpt_mngr


def load_train_ckpt(ckpt: str, state: TrainState):
    ckpt = Path(ckpt)

    train_ckpt = ckpt / "last"
    train_ckpt.mkdir(exist_ok=True)

    train_ckpt_mngr = ocp.CheckpointManager(
        train_ckpt,
        ocp.PyTreeCheckpointer(),
        ocp.CheckpointManagerOptions(max_to_keep=2, keep_period=1000),
    )
    return train_ckpt_mngr


def fit(
    hparams: HyperParams,
    ckpt: str,
    train_loader,
    val_loader,
    steps: int = 300_000,
    log_interval: int = 50,
    save_interval: int = 100,
    eval_interval: int = 10_0000,
    num_workers: int = 8,
):
    ckpt = Path(ckpt).absolute()
    hparam_path = ckpt / "hparams"

    hparam_ckptr = ocp.StandardCheckpointer()

    if hparam_path.exists():
        hparams = hparam_ckptr.restore(hparam_path, hparams)
    else:
        hparam_ckptr.save(hparam_path, hparams)

    key = random.PRNGKey(hparams.seed)
    state = create_state(key, hparams)

    train_ckpt_mngr = load_train_ckpt(ckpt, state)
    # best_ckpt_mngr = load_best_ckpt(ckpt)

    step = train_ckpt_mngr.latest_step()

    if step is not None:
        state = train_ckpt_mngr.restore(train_ckpt_mngr.latest_step(), state)
        step += 1
    else:
        step = 0

    state = jax_utils.replicate(state)

    rng_keys = jax.random.split(key, jax.local_device_count())

    p_train_step = jax.pmap(train_step, axis_name="device", donate_argnums=(0,))
    # p_eval_step = jax.pmap(eval_step, axis_name="device", donate_argnums=(0,))

    for batch in tqdm(train_loader, initial=step, dynamic_ncols=True):
        batch = common_utils.shard(batch)

        logs, state = p_train_step(state, batch, rng_keys)

        if step % log_interval == 0:
            logs = jax_utils.unreplicate(logs)

            losses = logs["lossees"]
            for name, metric in losses.itmes():
                wandb.log({f"train/{name}", metric.item(), step})

        # save train checkpoint
        if step % save_interval == 0:
            train_ckpt_mngr.save(step, jax_utils.unreplicate(state))

        """
        # eval loop
        metrics = {}
        if step % eval_interval == 0:
            for batch in tqdm(val_loader, position=1, leave=False, dynamic_ncols=True):
                batch = common_utils.shard(batch)

                logs = p_eval_step(state, batch, rng_key=rng_keys)

            best_ckpt_mngr.save(
                step,
                jax_utils.unreplicate(state),
                metrics=tree_util.tree_map(lambda x: x.item(), metrics),
            )
        """

        step += 1
