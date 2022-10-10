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

convention = "pyscf"

db = connect('geom_scan_200_sto_6g.db')
database = HamiltonianDatabase("gs_200_sto_6g.db", flags=(apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE))

cursor = database._get_connection().cursor()

for row in db.select(1):
    Zref = row['numbers']
    database.add_Z(Z=Zref)

# orbital ref for minimal basis set
orbitals_ref = {}
orbitals_ref[1] = np.array([0])       # H -> 1s
orbitals_ref[6] = np.array([0, 0, 1]) # C -> 1s, 2s, 2p

#add orbitals to database
for Z in database.Z:
    database.add_orbitals(Z, orbitals_ref[Z])

cursor.execute('''BEGIN''') #begin transaction
for i, row in enumerate(db.select()):
    if i % 100 == 0:
        print(i)
    Z = row['numbers']
    assert np.all(Z == Zref)
    R = row['positions'] * 1.8897261258369282 # convert angstrom to bohr
    # E = row.data['energy']
    # F = row.data['forces']
    H = row.data['F'].reshape(36, 36)
    # S = transform_hamiltonians.transform(row.data['overlap'],     atom_types, convention=convention)
    database.add_data( R=R, E=None, F=None, H=H, S=None, C=None, transaction=False)

cursor.execute('''COMMIT''') #commit transaction