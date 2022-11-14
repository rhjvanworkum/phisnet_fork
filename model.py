import pytorch_lightning as pl
import torch
from torchmetrics import MeanSquaredError

from torch.optim.lr_scheduler import _LRScheduler
class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class PhisNet(pl.LightningModule):
  
  def __init__(self, model, args) -> None:
    super().__init__()
    
    self.model = model
    self.args = args
    
    self.loss_keys = ['full_hamiltonian', 'overlap_matrix']
    self.mse = MeanSquaredError()
    # self.grad_enabled = len(self.model.required_derivatives) > 0
    
  def forward(self, inputs):
    results = self.model(R=inputs['positions'])
    return results
  
  def loss_fn(self, pred, targets):
    loss = 0.0
    batch_size = pred[self.loss_keys[0]].shape[0]
    bs_size = pred[self.loss_keys[0]].shape[-1]
    for key in self.loss_keys:
      for i in range(batch_size):
         loss += torch.sum(torch.square(targets[key][i].flatten() - pred[key][i].flatten())) / (bs_size ** 2)
    return loss / (len(self.loss_keys) * batch_size)
  
  def log_metrics(self, targets, pred, flag):
    for key in self.loss_keys:
      metric = self.mse(pred[key], targets[key])
      self.log(
        f"{flag}_{key}_mse",
        metric,
        on_step=False,
        on_epoch=True,
        prog_bar=False
      )
  
  def training_step(self, batch, batch_idx):
      targets = batch
      pred = self(batch)
      loss = self.loss_fn(pred, targets)
      self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
      self.log_metrics(targets, pred, "train")
      return loss

  def validation_step(self, batch, batch_idx):
      torch.set_grad_enabled(True)
      targets = batch
      pred = self(batch)
      loss = self.loss_fn(pred, targets)
      self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
      self.log_metrics(targets, pred, "val")
      return {"val_loss": loss}

  def test_step(self, batch, batch_idx):
      targets = batch
      pred = self(batch)
      loss = self.loss_fn(pred, targets)
      self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
      self.log_metrics(targets, pred, "test")
      return {"test_loss": loss}

  
  def configure_optimizers(self):
    parameters = []
    weight_decay_parameters = []
    for name, param in self.model.named_parameters():
        if 'weight' in name and not 'radial_fn' in name and not 'embedding' in name:
            weight_decay_parameters.append(param)
        else:
            parameters.append(param)

    parameter_list = [
        {'params': parameters},
        {'params': weight_decay_parameters, 'weight_decay': float(self.args.weight_decay)}
    ]
  
    optimizer = torch.optim.AdamW(parameter_list,  lr=self.args.learning_rate, eps=self.args.epsilon, betas=(self.args.beta1, self.args.beta2), weight_decay=0.0, amsgrad=True)

    schedulers = []
    # schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.args.decay_factor, patience=self.args.decay_patience)
    schedule = NoamLR(optimizer, warmup_steps=25)
    optimconf = {"scheduler": schedule, "name": "lr_schedule", "monitor": "val_loss"}
    schedulers.append(optimconf)
    return [optimizer], schedulers
  

  