import os
from pathlib import Path
from pprint import pprint
from typing import Dict

import copy

import monai
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
import json
from accelerate import Accelerator
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader, get_tbad_lab_unlab_transforms, load_dataset_images
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.SlimUNETR.SlimUNETR import SlimUNETR
from monai.networks.nets import UNet
from src.utils import Logger, load_config, same_seeds
from main_unlab_unet import get_experiment_dir


def calc_metrics_dict(metrics, accelerator, data_flag, is_train=True, unlab=False):
    metrics_dict = {}
    mode = "Train" if is_train else "Val"
    if unlab:
        mode = "Unlab_" + mode

    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        # print(batch_acc)
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
        elif data_flag in ["acute", "lung", "lung_big_model", "aneurysms"]:  # , "tbad_dataset"]:
            metrics_dict.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc),
                }
            )
        elif data_flag == "tbad_dataset":
            metrics_dict.update(
                {
                    f"{mode}/all {metric_name}": float(batch_acc[0]),
                    f"{mode}/TL {metric_name}": float(batch_acc[1]),
                    f"{mode}/FL {metric_name}": float(batch_acc[2]),
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

    return batch_acc, metrics_dict


def get_center_mask(np_mask):
    coord = np.where(np_mask == 1.)
    coord = np.stack((coord[0], coord[1], coord[2])).T
    return coord.mean(0)


def save_predict(data_batch, pred, path):

    path = Path(path)
    sample_name = Path(data_batch['label_meta_dict']['filename_or_obj'][0]).stem[:-4]

    img = np.array(data_batch["image"][0].cpu())
    lab = np.array(data_batch["label"][0].cpu())
    out = np.array(pred[0].cpu())

    nii = np.stack((lab[0], out[0])).transpose(1, 2, 3, 0)
    nii = nib.Nifti1Image(nii, affine=np.array(data_batch['label_meta_dict']['affine'][0].cpu()))
    
    if lab.sum() > 0:
        lab_c = get_center_mask(lab)
        out_c = get_center_mask(out)
        dist = round(np.abs(lab_c - out_c).mean(), 3) 
    else:
        dist = -1

    nib.save(nii, path / f'lab_{sample_name}_{np.sum(lab)}_{np.sum(out)}_metr{round(1. - float(np.sum(np.abs(lab - out)) / (max(np.sum(lab), np.sum(out))) + 1e-6), 3)}_dist{dist}.nii.gz')

    nii = nib.Nifti1Image(img[0], affine=np.array(data_batch['label_meta_dict']['affine'][0].cpu()))
    nib.save(nii, path / f'img_{sample_name}.nii.gz')



@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    data_flag: str,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch,
    path,
    device
):
    
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model.eval()
    for data_batch in tqdm(val_loader):
        logits = inference(data_batch["image"].to(device), model)
        val_outputs = [post_trans(i) for i in logits]
 
        save_predict(data_batch, val_outputs, path)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=data_batch["label"].to(device))
  
    batch_acc, metrics_dict = calc_metrics_dict(
        metrics, accelerator, data_flag, is_train=False
    )

    return batch_acc, metrics_dict


def load_sampels(config):
    # load without shuffle (save map index and pat №)
    
    data_list = load_dataset_images(config.data_root)
    _, _, val_transform = get_tbad_lab_unlab_transforms(config)
    loader = monai.data.DataLoader(
        dataset=monai.data.Dataset(
            data=data_list,
            transform=val_transform,
        ),
        num_workers=config.trainer.num_workers,
        batch_size=1,
        shuffle=False,
    )

    data_list = [
        item['image'].split('_')[-1].split('.')[0]
        for item in data_list
    ]
    
    return loader, data_list



if __name__ == "__main__":

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    config, data_flag, is_HepaticVessel = load_config(
        config_filename="config.yml", mode="r"
    )

    same_seeds(config.trainer.seed)
    logging_dir = get_experiment_dir(config, data_flag)

    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    Logger(logging_dir)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    path = Path(f'./{data_flag}/save_eval_slim_tbda')
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.json").open("w") as fp:
        json.dump(config , fp)
    

    accelerator.print("Load Model...")
    model = SlimUNETR(**config.slim_unetr)
    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=1,
    #     channels=(24, 48, 60),
    #     strides=(2, 1),
    #     dropout=0.3,
    # )

    accelerator.print("Load Dataloader...")
    config_copy = copy.copy(config)

    train_loader, val_loader, unlab_loader = get_dataloader(config, data_flag, needs_unlab=True)
    print(unlab_loader)
    loader, data_list = load_sampels(config)

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
    )
    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }

    step = 0
    best_epoch = -1
    val_step = 0
    starting_epoch = 0
    best_acc = 0
    best_class = []

    model, optimizer, scheduler, loader = accelerator.prepare(
        model, optimizer, scheduler, loader
    )

    base_exp_path_save = get_experiment_dir(config, data_flag, root="model_store", model='')

    print(base_exp_path_save)
    print(sum(p.numel() for p in model.parameters()))

    list_epochs = list(base_exp_path_save.iterdir())[1:] 

    print(list_epochs)

    for epoch_folder in sorted(list_epochs, key=lambda x: int(x.name.split('_')[-1]))[1:]:

        epoch = int(epoch_folder.name.split('_')[-1])

        model = utils.load_pretrain_model(
            str(epoch_folder / "pytorch_model.bin"),
            model,
            accelerator,
        )

        device = next(model.parameters()).device

        # val
        val_metric, metrics_dict = val_one_epoch(
            model,
            data_flag,
            inference,
            loader,
            metrics,
            post_trans,
            accelerator,
            epoch=epoch,
            path=f'./{data_flag}/save_eval_slim_/epoch_{epoch}',
            device=device
        )


        unlab_list = [] 
        for data_batch in tqdm(unlab_loader):
            unlab_list.extend(data_batch['label_meta_dict']['filename_or_obj'])
        
        pprint(unlab_list)
            
        accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] metric {metrics_dict}"
    )
