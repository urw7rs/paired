import numpy as np
from scipy.spatial.transform import Rotation as R


smpl_parents = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]

offsets = np.array(smpl_offsets)
parents = np.array(smpl_parents)

has_children = np.zeros(len(parents)).astype(bool)
for parent in parents:
    if parent != -1:
        has_children[parent] = True

children = [[] for _ in range(len(parents))]

for i, parent in enumerate(parents):
    if parent != -1:
        children[parent].append(i)


def fk(pose, trans):
    assert len(pose.shape) == 3
    assert pose.shape[1] == 24
    assert len(trans.shape) == 2

    world_positions = []
    world_rotations = []

    # Parallelize along the batch and time dimensions
    for i, parent in enumerate(parents):
        if parent == -1:
            world_positions.append(trans)
            r = R.from_rotvec(pose[:, 0])
            world_rotations.append(r)
        else:
            pos = world_rotations[parent].apply(offsets[i]) + world_positions[parent]
            world_positions.append(pos)

            if has_children[i]:
                r = R.from_rotvec(pose[:, i])
                world_rotations.append(world_rotations[parent] * r)
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                world_rotations.append(None)

    return np.stack(world_positions, axis=1)
