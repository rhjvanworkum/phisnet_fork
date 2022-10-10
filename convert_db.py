#!/usr/bin/env python3
import numpy as np
from base64 import b64decode
from ase.db import connect
import apsw
import transform_hamiltonians
from training.sqlite_database import HamiltonianDatabase

convention = "orca_def2-SVP"

db = connect('schnorb_hamiltonian_water.db')
database = HamiltonianDatabase("h2o_pbe-def2svp_4999.db", flags=(apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE))

cursor = database._get_connection().cursor()

chemical_symbols = ['n', 'H', 'He','Li', 'Be', 'B', 'C', 'N', 'O']

for row in db.select(1):
    Zref = row['numbers']
    atom_types = ''.join([chemical_symbols[i] for i in Zref])
    database.add_Z(Z=Zref)

#orbital reference (for def2svp basis)
orbitals_ref = {}
orbitals_ref[1] = np.array([0,0,1])       #H: 2s 1p
orbitals_ref[6] = np.array([0,0,0,1,1,2]) #C: 3s 2p 1d
orbitals_ref[7] = np.array([0,0,0,1,1,2]) #N: 3s 2p 1d
orbitals_ref[8] = np.array([0,0,0,1,1,2]) #O: 3s 2p 1d

#add orbitals to database
for Z in database.Z:
    database.add_orbitals(Z, orbitals_ref[Z])

#def decode(row, data):
#    print(row.data)
#    quit()
#    shape = row.data['_shape_'+data]
#    dtype = row.data['_dtype_'+data]
#    if np.sum(shape) > 0:
#        return np.frombuffer(b64decode(row.data[data]), dtype=dtype).reshape(shape)
#    else:
#        return None

cursor.execute('''BEGIN''') #begin transaction
for i, row in enumerate(db.select()):
    if i % 100 == 0:
        print(i)
    Z = row['numbers']
    assert np.all(Z == Zref)
    R = row['positions']*1.8897261258369282 #convert angstrom to bohr
    E = row.data['energy']
    F = row.data['forces']
    H = transform_hamiltonians.transform(row.data['hamiltonian'], atom_types, convention=convention)
    S = transform_hamiltonians.transform(row.data['overlap'],     atom_types, convention=convention)
    database.add_data( R=R, E=E, F=F, H=H, S=S, C=None, transaction=False)
cursor.execute('''COMMIT''') #commit transaction
