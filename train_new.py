import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
import logging
from datamodule import CustomDataModule
from model import PhisNet

from training import *
from nn import *

WAND_PROJECT = 'phisnet'

def load_model(args, dataset, use_gpu):
  model =  NeuralNetwork(
        orbitals             = dataset.orbitals,
        order                = args.order,
        num_features         = args.num_features,
        num_basis_functions  = args.num_basis_functions,
        num_modules          = args.num_modules,
        num_residual_pre_x   = args.num_residual_pre_x,
        num_residual_post_x  = args.num_residual_post_x,
        num_residual_pre_vi  = args.num_residual_pre_vi,
        num_residual_pre_vj  = args.num_residual_pre_vj,
        num_residual_post_v  = args.num_residual_post_v,
        num_residual_output  = args.num_residual_output,
        num_residual_pc      = args.num_residual_pc,
        num_residual_pn      = args.num_residual_pn,
        num_residual_ii      = args.num_residual_ii,
        num_residual_ij      = args.num_residual_ij,
        num_residual_full_ii = args.num_residual_full_ii,
        num_residual_full_ij = args.num_residual_full_ij,
        num_residual_core_ii = args.num_residual_core_ii,
        num_residual_core_ij = args.num_residual_core_ij,
        num_residual_over_ij = args.num_residual_over_ij,
        basis_functions      = args.basis_functions,
        cutoff               = args.cutoff,
        activation           = args.activation)
  
  model.to(args.dtype)
  
  if use_gpu:
    model = model.cuda()
  
  return model

if __name__ == "__main__":
  args = parse_command_line_arguments()
  use_wandb = True
  use_gpu = torch.cuda.is_available()
  name = 'fulvene_s01_200_F_new_2'

  datamodule = CustomDataModule(args)

  model = load_model(args, datamodule.dataset, use_gpu)
  phisnet = PhisNet(model=model, args=args)
  phisnet.model.calculate_full_hamiltonian = True
  phisnet.model.calculate_core_hamiltonian = False
  phisnet.model.calculate_overlap_matrix = True
  phisnet.model.calculate_energy = True
  phisnet.model.calculate_forces = False
  
  callbacks = [
    pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="step"),
    pytorch_lightning.callbacks.ModelCheckpoint(
      monitor='val_loss',
      dirpath='./checkpoints/',
      filename=name + '-{epoch:02d}-{val_loss:.2f}',
      mode = 'min',
    )
  ]
  
  if use_wandb:
    logger = WandbLogger(project=WAND_PROJECT)
    trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                        logger=logger,
                                        default_root_dir='./test/',
                                        max_epochs=1000,
                                        accelerator='gpu',
                                        # deterministic=True,
                                        devices=1)
  else:
    trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                    default_root_dir='./test/',
                                    max_epochs=1000,
                                    accelerator='gpu',
                                    # deterministic=True,
                                    devices=1) 
  
  logging.info("Start training")
  trainer.fit(phisnet, datamodule=datamodule)