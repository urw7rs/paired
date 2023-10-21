import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

from .skeleton import smpl_parents


def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact):
    pose = poses[num]
    static = contact[num]
    indices = [7, 8, 10, 11]

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
        color = "r" if static[i] else "g"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 2.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)


def plot_skeleton(gifname, poses, contact=None, fps: int = 60):
    num_steps = poses.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    point = np.array([0, 0, 1])
    normal = np.array([0, 0, 1])
    d = -point.dot(normal)
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
    # plot the plane
    ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)
    # Create lines initially without data
    lines = [ax.plot([], [], [], zorder=10, linewidth=1.5)[0] for _ in smpl_parents]
    scat = [
        ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
        for _ in range(4)
    ]
    axrange = 3

    # create contact labels
    feet = poses[:, (7, 8, 10, 11)]
    feetv = np.zeros(feet.shape[:2])
    feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
    if contact is None:
        contact = feetv < 0.01
    else:
        contact = contact > 0.95

    # Creating the Animation object
    anim = animation.FuncAnimation(
        fig,
        plot_single_pose,
        num_steps,
        fargs=(poses, lines, ax, axrange, scat, contact),
        interval=0,
    )

    anim.save(
        gifname, savefig_kwargs={"transparent": True, "facecolor": "none"}, fps=fps
    )
    plt.close()
