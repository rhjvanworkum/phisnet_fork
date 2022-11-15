import numpy as np
from argparse import Namespace

# definition of AO's for molecules in a specific basis
orbital_definitions = {
    'fulvene_minimal_basis': {
        1: np.array([0]),           # H has only 1s orbital
        6: np.array([0, 0, 1])      # C has 1s, 2s & 2p orbital
    }
}

# definition of how AO's are outputted by different quantum chemistry programs in different basis sets
orbital_conventions = {
    """ Taken from original PhisNet implementation """
    'orca_631Gss': Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sspspd', 'C': 'sspspd', 'N': 'sspspd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 3, 2, 4, 5], 'C': [0, 1, 3, 2, 4, 5], 'N': [0, 1, 3, 2, 4, 5]},
    ),
    'orca_def2-SVP': Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    ),
    'aims': Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [0, 1, 2], 'd': [0, 1, 2, 3, 4]},
        orbital_sign_map={'s': [1], 'p': [1, 1, -1], 'd': [1, 1, 1, -1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5]},
    ),
    'psi4': Namespace(
        atom_to_orbitals_map={'H': 'ssp', 'O': 'sssppd', 'C': 'sssppd', 'N': 'sssppd', 'F': 'sssppd'},
        orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [4, 2, 0, 1, 3]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]},
        orbital_order_map={'H': [0, 1, 2], 'O': [0, 1, 2, 3, 4, 5], 'C': [0, 1, 2, 3, 4, 5], 'N': [0, 1, 2, 3, 4, 5], 'F': [0, 1, 2, 3, 4, 5]},
    ),
    """ Added by us - all apply to PySCF """
    'fulvene_minimal_basis': Namespace(
        atom_to_orbitals_map={'H': 's', 'C': 'ssp'},
        orbital_idx_map={'s': [0], 'p': [1, 2, 0]},
        orbital_sign_map={'s': [1], 'p': [1, 1, 1]},
        orbital_order_map={'H': [0], 'C': [0, 1, 2, 3, 4]},
    ),
}