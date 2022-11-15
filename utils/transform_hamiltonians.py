import numpy as np
from typing import List

from utils.definitions import orbital_conventions


def transform_hamiltonians_from_ao_to_lm(hamiltonians: np.ndarray, atoms: List[str], convention: str) -> np.ndarray:
    """
    Transforms the ordering of elements in the hamiltonian from the AO ordering to the
    ordering of (l, m) as used in the spherical harmonics.
    """
    conv = orbital_conventions[convention]

    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append((np.array(map_idx) + offset).tolist())
        transform_signs.append(map_sign)
    transform_indices = [item for sublist in transform_indices for item in sublist]
    transform_signs = [item for sublist in transform_signs for item in sublist]

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.array(transform_indices).astype(np.int)
    transform_signs = np.array(transform_signs)

    hamiltonians_new = hamiltonians[...,transform_indices, :]
    hamiltonians_new = hamiltonians_new[...,:, transform_indices]
    hamiltonians_new = hamiltonians_new * transform_signs[:, None]
    hamiltonians_new = hamiltonians_new * transform_signs[None, :]

    return hamiltonians_new


# def transform_back(hamiltonians: np.ndarray, convention: str) -> np.ndarray:
#     if convention == 'aims':
#         transformed_hamiltonians, nonzero_indices = transform_to_aims(hamiltonians)
#     if convention == 'orca':
#         transformed_hamiltonians, nonzero_indices = transform_to_orca(hamiltonians)
#     return transformed_hamiltonians, nonzero_indices

# def transform_to_aims(hamiltonians):
#     hamiltonians = np.transpose(hamiltonians, axes=(1, 2, 0))  # j, i, batch

#     hamiltonians_new = np.zeros((3 * 14, 3 * 14, hamiltonians.shape[2]))
#     mapping = [
#         (np.arange(14), np.arange(14)),
#         (np.array([14, 15, 17, 18, 19]), np.arange(14, 14 + 5)),
#         (np.array([28, 29, 31, 32, 33]), np.arange(14 + 5, 14 + 5 + 5))
#     ]
#     for i_out, i_in in mapping:
#         for j_out, j_in in mapping:
#             print(np.meshgrid(i_out, j_out))
#             hamiltonians_new[tuple(np.meshgrid(i_out, j_out))] = hamiltonians[tuple(np.meshgrid(i_in, j_in))]
#     hamiltonians_new = np.transpose(hamiltonians_new, axes=(2, 0, 1))  # batch, i, j
#     nonzero_indices = np.concatenate([out for out, _in in mapping])

#     # Change of signs
#     # (1, 1, -1) for l=1
#     # (1, 1, 1, -1, 1) for l=2
#     hamiltonians_new = hamiltonians_new.reshape(-1, 3, 14, 3, 14)
#     for i in [5, 8, 12]:
#         hamiltonians_new[:, :, i, :, :] = -hamiltonians_new[:, :, i, :, :]
#         hamiltonians_new[:, :, :, :, i] = -hamiltonians_new[:, :, :, :, i]
#     hamiltonians_new = hamiltonians_new.reshape(-1, 3 * 14, 3 * 14)
#     hamiltonians_new = hamiltonians_new[:, nonzero_indices][:, :, nonzero_indices]

#     return hamiltonians_new, nonzero_indices

# def transform_to_orca(hamiltonians):
#     hamiltonians = np.transpose(hamiltonians, axes=(1, 2, 0))  # j, i, batch
#     # hamiltonians[2:6, :, :] = hamiltonians[[5, 2, 3, 4], :, :]
#     # hamiltonians[:, 2:6, :] = hamiltonians[:, [5, 2, 3, 4], :]

#     hamiltonians_new = np.zeros((24, 24, hamiltonians.shape[2]))
#     mapping = [
#         (np.arange(2), np.arange(2)),
#         (np.arange(2, 6), np.array([4, 5, 3, 2])),  # move s orbital and rearange p orbitals
#         (np.arange(6, 9), np.array([7, 8, 6])),  # rearrange p orbitals
#         (np.arange(9, 14), np.array([11, 12, 10, 13, 9])),  # rearrange d orbitals
#         (np.arange(14, 19), np.array([14, 15, 17, 18, 16])),
#         (np.arange(19, 24), np.array([19, 20, 22, 23, 21]))
#     ]
#     for i_out, i_in in mapping:
#         for j_out, j_in in mapping:
#             print(np.meshgrid(i_out, j_out))
#             hamiltonians_new[tuple(np.meshgrid(i_out, j_out))] = hamiltonians[tuple(np.meshgrid(i_in, j_in))]
#     hamiltonians_new = np.transpose(hamiltonians_new, axes=(2, 0, 1))  # batch, i, j
#     nonzero_indices = np.concatenate([out for out, _in in mapping])

#     hamiltonians_new = hamiltonians_new[:, nonzero_indices][:, :, nonzero_indices]

#     return hamiltonians_new, nonzero_indices
