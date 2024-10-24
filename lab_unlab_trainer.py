import copy

import torch
from torch import nn

from src.unlab.transforms import Transforms


class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.96, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def eval(self):
        self.module.eval()

    def forward(self, x):
        return self.module(x)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

def calc_total_loss(logits, label, loss_functions, accelerator, step, train=True):
    log = ""
    total_loss = 0
    name_stage = "Train" if train else "Val"
    for name in loss_functions:
        loss_fn, ratio = loss_functions[name]
        loss = ratio * loss_fn(logits, label)
        accelerator.log({f"{name_stage}/" + name: float(loss)}, step=step)
        log += f" {name} {float(loss):1.5f} "
        total_loss += loss

    return total_loss, log


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        unlab_loader,
        optimizer,
        accelerator,
        loss_functions,
        post_trans,
        cfg,
    ):

        self.model = model
        self.ema_model = ModelEmaV2(model)
        self.train_loader = train_loader
        self.unlab_loader = unlab_loader

        self.optimizer = optimizer
        self.accelerator = accelerator
        self.loss_functions = loss_functions
        self.post_trans = post_trans
        self.transforms = Transforms(
            rot_prob=cfg.rot_prob,
            flip_prob=cfg.flip_prob,
            rot_range_z=cfg.rot_angle,
            final_image_size=cfg.image_size
        )
        self.unlab_weight = cfg.unlab_weight

    def train_labeled_one_epoch(
        self,
        metrics,
        num_epochs: int,
        epoch: int,
        step: int,
        use_transform: bool
    ):
        device = next(self.model.parameters()).device
        self.model.train()
        for i, image_batch in enumerate(self.train_loader):
            image = image_batch["image"].to(device)
            label = image_batch["label"].to(device)

            logits = self.model(image)
            total_loss, _ = calc_total_loss(
                logits, label, self.loss_functions, self.accelerator, step
            )

            if use_transform:
                image_tf = self.transforms(image)
                logits_tf = self.model(image_tf)
                label_tf = self.transforms(label, randomize=False)

                tf_loss, _ = calc_total_loss(
                    logits_tf, label_tf, self.loss_functions, self.accelerator, step
                )
                total_loss += tf_loss


            dict_values = {"Train/Total Loss": float(total_loss)}
            log =  f"Epoch [{epoch + 1}/{num_epochs}] Training [{i + 1}/{len(self.train_loader)}] Loss: {total_loss:1.5f}"
            if use_transform:
                dict_values.update(
                    {"Train/TrFm Loss": float(tf_loss)}
                )
                log += f" TrFm Loss: {tf_loss:1.5f}"

            self.accelerator.log(values=dict_values, step=step)
            self.accelerator.print(log, flush=True)
            step += 1

            self.accelerator.backward(total_loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            val_outputs = [self.post_trans(i) for i in logits]
            for metric_name in metrics:
                metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])

            return step

