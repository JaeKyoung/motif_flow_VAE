import pytorch_lightning as pl
import copy
from pytorch_lightning.callbacks import Callback
import torch

class LAWACallback(Callback):
    """
    LAWA (Latest Weight Averaging) Callback.
    
    Following the paper's pseudocode: https://arxiv.org/pdf/2209.14981
    Supports two modes:
      - Uniform averaging: stores state_dict in a queue and averages the last k checkpoints.
      - EMA mode: applies exponential moving average (EMA) update with decay_rate.
    
    The averaging (or EMA update) only starts after `lawa_start_epoch`.
    At training end, the LAWA model's weights are applied to the final model.
    """
    def __init__(self, k=6, device=None, use_ema=False, decay_rate=0.9, lawa_start_epoch=10):
        """
       Args:
            k (int): Number of recent checkpoints to average (default k=6 as recommended).
            device (torch.device, optional): Device to store checkpoints. Defaults to CPU.
            use_ema (bool): If True, uses EMA update rule instead of uniform averaging.
            decay_rate (float): EMA decay rate (if use_ema is True). Typical values around 0.9.
            lawa_start_epoch (int): Epoch from which to start collecting checkpoints/EMA updates.
        """
        super().__init__()
        self.k = k
        self.device = device or torch.device('cpu')
        self.use_ema = use_ema
        self.decay_rate = decay_rate
        self.lawa_start_epoch = lawa_start_epoch
        
        # For uniform averaging: queue for storing checkpoints.
        self.ckpts = []
        # LAWA model that will be updated either via uniform averaging or EMA.
        self.lawa_model = None

    def on_train_start(self, trainer, pl_module):
        # Initialize LAWA model by deep copying the initial model state
        self.lawa_model = copy.deepcopy(pl_module)
        self.lawa_model.to(self.device)
    
    def _uniform_average(self):
        """
        Compute averaged state_dict from stored checkpoints (uniform averaging).
        Returns:
            dict: Averaged state dictionary.
        """
        avg_state_dict = {}
        for key in self.ckpts[0]:
            # Stack tensors along new dimension for efficient averaging
            stacked_params = torch.stack([ckpt[key] for ckpt in self.ckpts])
            avg_state_dict[key] = torch.mean(stacked_params, dim=0)
        return avg_state_dict

    def _ema_update(self, current_state):
        """
        Update LAWA model's state using EMA update rule:
          new_state = decay_rate * old_state + (1-decay_rate) * current_state.
        """
        lawa_state = self.lawa_model.state_dict()
        for key in lawa_state:
            lawa_state[key] = self.decay_rate * lawa_state[key] + (1 - self.decay_rate) * current_state[key]
        self.lawa_model.load_state_dict(lawa_state)
    
    def on_epoch_end(self, trainer, pl_module):
        # LAWA update only starts after lawa_start_epoch.
        if trainer.current_epoch < self.lawa_start_epoch:
            return
        
        # Obtain current model state (detached and moved to specified device)
        current_state = {
            key: param.detach().clone().to(self.device)
            for key, param in pl_module.state_dict().items()
        }
        if self.use_ema:
            # EMA mode: update LAWA model using EMA update rule.
            self._ema_update(current_state)
        else:
            # Uniform averaging mode: add current state to queue.
            self.ckpts.append(current_state)
            # If we have collected at least k checkpoints, compute average and update.
            if len(self.ckpts) >= self.k:
                avg_state_dict = self._uniform_average()
                self.lawa_model.load_state_dict(avg_state_dict)
                # Remove the oldest checkpoint to maintain queue size.
                self.ckpts.pop(0)
        
        torch.cuda.empty_cache()
        
    def on_train_end(self, trainer, pl_module):
        # At training end, apply LAWA model's weights to the final model.
        if self.lawa_model is not None:
            pl_module.load_state_dict(self.lawa_model.state_dict())
            print("LAWA: Successfully applied final averaged weights to model")
        self.ckpts.clear()
        self.lawa_model = None
        torch.cuda.empty_cache()