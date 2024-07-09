import os
from pathlib import Path
import sys
from typing import Dict
from easydict import EasyDict

from objprint import objstr

import torch
from torch import nn
import monai
from monai.utils import ensure_tuple_rep
from accelerate import Accelerator
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.utils import Logger, load_config, same_seeds


def calc_total_loss(logits, label, loss_functions):
    log = ""
    total_loss = 0
    for name in loss_functions:
        loss = loss_functions[name](logits, label)
        accelerator.log({"Train/" + name: float(loss)}, step=step)
        log += f" {name} {float(loss):1.5f} "
        total_loss += loss

    return total_loss, log


def calc_metrics_dict(metrics, accelerator, data_flag, is_train=True):
    metrics_dict = {}
    mode = "Train" if is_train else "Val"
    print(metrics.keys())
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        print(batch_acc)
        if accelerator.num_processes > 1:
            batch_acc = (
                accelerator.reduce(batch_acc.to(accelerator.device))
                / accelerator.num_processes
            )
        metrics[metric_name].reset()

        metrics_dict[f"{mode}/mean {metric_name}"] = float(batch_acc.mean())
        if data_flag == "hepatic_vessel2021":
            metrics_dict.update(
                {
                    f"{mode}/Hepatic Vessel {metric_name}": float(batch_acc[0]),
                    f"{mode}/Tumors {metric_name}": float(batch_acc[1]),
                }
            )
        elif data_flag in ["acute", "lung", "lung_big_model"]:  # , "tbad_dataset"]:
            metrics_dict.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc),
                }
            )
        elif data_flag == "tbad_dataset":
            metrics_dict.update(
                {
                    f"{mode}/TL {metric_name}": float(batch_acc[0]),
                    f"{mode}/FL {metric_name}": float(batch_acc[1]),
                    f"{mode}/FLT {metric_name}": float(batch_acc[2]),
                }
            )
        else:
            metrics_dict.update(
                {
                    f"{mode}/TC {metric_name}": float(batch_acc[0]),
                    f"{mode}/WT {metric_name}": float(batch_acc[1]),
                    f"{mode}/ET {metric_name}": float(batch_acc[2]),
                }
            )

    return metrics_dict, batch_acc


def regularization_loss(hidden_states_out):
    pass


def train_one_epoch(
    model: torch.nn.Module,
    config: EasyDict,
    data_flag: str,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    # train
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch["image"])
        # print(logits.shape)
        # print(image_batch["label"].shape)
        total_loss, _ = calc_total_loss(logits, image_batch["label"], loss_functions)

        accelerator.log(values={"Train/Total Loss": float(total_loss)}, step=step)
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss: {total_loss:1.5f}",
            flush=True,
        )
        step += 1

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()

        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])

    scheduler.step(epoch)
    metric, _ = calc_metrics_dict(metrics, accelerator, data_flag, is_train=True)
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    return step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    data_flag: str,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    config: EasyDict,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
):
    # val
    device = next(model.parameters()).device
    print(device)
    model.eval()
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch["image"].to(device), model)
        total_loss = 0
        log = ""
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch["label"].to(device))
            accelerator.log({"Val/" + name: float(loss)}, step=step)
            log += f" {name} {float(loss):1.5f} "
            total_loss += loss
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        accelerator.log(
            {
                "Val/Total Loss": float(total_loss),
            },
            step=step,
        )
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}",
            flush=True,
        )
        step += 1

    metric, batch_acc = calc_metrics_dict(
        metrics, accelerator, data_flag, is_train=False
    )

    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}"
    )
    print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}")
    accelerator.log(metric, step=epoch)
    return (
        torch.Tensor([metric["Val/mean dice_metric"]]).to(accelerator.device),
        batch_acc,
        step,
    )


def get_logging_dir(config, data_flag):
    logging_dir = Path(os.getcwd()) / "logs"

    logging_dir /= data_flag

    logging_dir /= f"seed{config.trainer.seed}"

    logging_dir /= f"epoch{config.trainer.num_epochs}"

    logging_dir /= (
        f"ims_{config.trainer.image_size}_rot_prob{config.trainer.rot_prob}_leaky_relu"
    )

    logging_dir.mkdir(parents=True, exist_ok=True)
    return logging_dir


if __name__ == "__main__":

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    config, data_flag, is_HepaticVessel = load_config(
        config_filename="config.yml", mode="r"
    )

    same_seeds(config.trainer.seed)
    logging_dir = get_logging_dir(config, data_flag)

    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=str(logging_dir)
    )
    Logger(logging_dir)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("Load Model...")
    model = SlimUNETR(**config.slim_unetr)

    accelerator.print("Load Dataloader...")
    train_loader, val_loader = get_dataloader(config, data_flag)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(config.trainer.image_size, dim=3),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
        # 'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True, reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=False)
    }
    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=config.trainer.weight_decay,
        lr=config.trainer.lr,
        betas=(0.9, 0.95),
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
        eta_min=config.trainer.min_lr,
    )

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
        # "tversky_loss": monai.losses.TverskyLoss(
        #     to_onehot_y=False, sigmoid=True, alpha=0.4, beta=0.6
        # ),
    }

    step = 0
    best_eopch = -1
    val_step = 0
    starting_epoch = 0
    best_acc = 0
    best_class = []

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    def _weights_init(m):

        classname = m.__class__.__name__

        if isinstance(m, (nn.Conv3d, nn.Linear, nn.ConvTranspose3d)):
            nn.init.kaiming_uniform_(m.weight, a=10e-6)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find("Norm") != -1:
            m.weight.data.fill_(10e-7)
            nn.init.zeros_(m.bias)

    model.apply(_weights_init)

    base_exp_path = f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/seed{config.trainer.seed}/epoch{config.trainer.num_epochs}/ims_{config.trainer.image_size}_rot_prob{config.trainer.rot_prob}_leaky_relu"

    # resume training
    if config.trainer.resume:
        model, starting_epoch, step, val_step = utils.resume_train_state(
            model, base_exp_path, train_loader, accelerator
        )

    # Start Training
    accelerator.print("Start Training!")  # type: ignore
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # train
        step = train_one_epoch(
            model,
            config,
            data_flag,
            loss_functions,
            train_loader,
            optimizer,
            scheduler,
            metrics,
            post_trans,
            accelerator,
            epoch,
            step,
        )

        # val
        mean_acc, batch_acc, val_step = val_one_epoch(
            model,
            data_flag,
            loss_functions,
            inference,
            val_loader,
            config,
            metrics,
            val_step,
            post_trans,
            accelerator,
            epoch,
        )

        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] lr = {scheduler.get_last_lr()} best acc: {best_acc}, mean acc: {mean_acc}, mean class: {batch_acc}"
        )

        # save model
        if mean_acc > best_acc:
            accelerator.save_state(output_dir=f"{base_exp_path}/best")
            best_acc = mean_acc
            best_class = batch_acc
            best_eopch = epoch

        if epoch % 10 == 0:
            accelerator.save_state(output_dir=f"{base_exp_path}/epoch_{epoch}")

    accelerator.print(f"best dice mean acc: {best_acc}")
    accelerator.print(f"best dice accs: {best_class}")
    sys.exit(1)
