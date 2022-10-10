from ase.db import connect
from pyscf import gto
from pyscf.tools import molden
import numpy as np

from training import *
from nn import *
from datamodule import CustomDataModule
from model import PhisNet
from train_new import load_model


def write_db_entry_to_molden_file(molden_file,
                                  atom_numbers,
                                  atom_positions,
                                  mo_coeffs):

    atom_string = ""
    for atom, position in zip(atom_numbers, atom_positions):
      atom_string += f'{atom} {position[0]} {position[1]} {position[2]}; '

    molecule = gto.M(atom=atom_string,
                    basis='sto_6g',
                    spin=0,
                    symmetry=True)

    with open(molden_file, 'w') as f:
        molden.header(molecule, f)
        molden.orbital_coeff(molecule, f, mo_coeffs, ene=np.zeros(mo_coeffs.shape[0]))


def run_casscf_calculation(atom_numbers,
                           atom_positions,
                           guess_orbitals: np.ndarray,
                           basis='sto-6g'):
  
  atom_string = ""
  for atom, position in zip(atom_numbers, atom_positions):
    atom_string += f'{atom} {position[0]} {position[1]} {position[2]}; '

  molecule = gto.M(atom=atom_string,
                  basis='sto_6g',
                  spin=0,
                  symmetry=True)

  hartree_fock = molecule.RHF()
  n_states = 2
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  casscf.conv_tol = 1e-8

  e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(guess_orbitals)
  return imacro, imicro, iinner  


if __name__ == "__main__":
  args = parse_command_line_arguments()
  use_gpu = torch.cuda.is_available()
  if use_gpu:
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  
  datamodule = CustomDataModule(args)

  model = load_model(args, datamodule.dataset, use_gpu)
  phisnet = PhisNet(model=model, args=args)
  
  checkpoint = torch.load('checkpoints/test-epochepoch=99-val_lossval_loss=0.02.ckpt')
  phisnet.load_state_dict(checkpoint['state_dict'])
  phisnet.model.eval()
      
  with connect('geom_scan_200_sto_6g.db') as conn:
    R = conn.get(1)['positions'] * 1.8897261258369282 # convert angstroms to bohr
  
    input = {'positions': torch.stack([torch.tensor(R, dtype=torch.float32)]).to(device)}
    output = phisnet(input)
    # print(output['full_hamiltonian'])
    mo_coeffs = output['orbital_coefficients'][0].detach().cpu().numpy()
    # write_db_entry_to_molden_file('test.molden', conn.get(1)['numbers'], conn.get(1)['positions'], mo_coeffs)
    
    mo_coeffs = mo_coeffs.astype(np.double)
    results = run_casscf_calculation(conn.get(1)['numbers'], conn.get(1)['positions'], mo_coeffs)
    print(results)
  
  
# database = HamiltonianDatabase('gs_200_sto_6g.db')
# batch = [1]
# all_data = database[batch] #fetch the batch data
# R, E, F, H, S, C = [], [], [], [], [], []
# for batch_num, data in enumerate(all_data):
#     R_, E_, F_, H_, S_, C_ = data
#     R.append(torch.tensor(R_))
#     E.append(torch.tensor(E_))
#     F.append(torch.tensor(F_))
#     H.append(torch.tensor(H_))
#     S.append(torch.tensor(S_))
#     C.append(torch.tensor(C_))
  