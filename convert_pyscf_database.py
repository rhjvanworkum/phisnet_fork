"""
Script to convert PySCF orbital database to PhisNet format

(in minimal basis set for now)

PyScf outputs orbitals already in perfectly aligned order with atoms in DB.
Therefore the Fock matrix block 1x1 already corresponds to hydrogen 1 1s orbital interaction

So no need for tranformation back & forth
"""

#!/usr/bin/env python3
import numpy as np
from ase.db import connect
import apsw
import numpy as np
from training.sqlite_database import HamiltonianDatabase
from transform_hamiltonians import transform

convention = "pyscf"
read_db_name = "fulvene_s01_200.db"
write_db_name = "fulvene_s01_200_phisnet.db"
n = 36

db = connect(read_db_name)
database = HamiltonianDatabase(write_db_name, flags=(apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE))

cursor = database._get_connection().cursor()

for row in db.select(1):
    Zref = row['numbers']
    database.add_Z(Z=Zref)

# orbital ref for minimal basis set
orbitals_ref = {}
orbitals_ref[1] = np.array([0])       # H -> 1s
orbitals_ref[6] = np.array([0, 0, 1]) # C -> 1s, 2s, 2p

atoms = []
dict = {6: 'C', 1: 'H'}
#add orbitals to database
for Z in database.Z:
    database.add_orbitals(Z, orbitals_ref[Z])
    atoms.append(dict[Z])

# print(atoms)

cursor.execute('''BEGIN''') #begin transaction
for i, row in enumerate(db.select()):
    if i % 100 == 0:
        print(i)
    Z = row['numbers']
    assert np.all(Z == Zref)
    R = row['positions'] * 1.8897261258369282 # convert angstrom to bohr
    # E = row.data['energy']
    # F = row.data['forces']
    H = transform(row.data['F'].reshape(36, 36), atoms=atoms, convention='pyscf_minimal')
    S = transform(row.data['S'].reshape(36, 36), atoms=atoms, convention='pyscf_minimal')
    database.add_data( R=R, E=None, F=None, H=H, S=S, C=None, transaction=False)

cursor.execute('''COMMIT''') #commit transaction