import os
from pathlib import Path
from pprint import pprint
from typing import Dict

import copy

import monai
import numpy as np
import nibabel as nib
import torch
from torchinfo import summary

from tqdm import tqdm
import json
from accelerate import Accelerator
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader, get_tbad_lab_unlab_transforms, load_dataset_images
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.networks import create_model
from src.utils import Logger, load_config, same_seeds
from main_unlab_unet import get_new_experiment_dir


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
        elif data_flag in ["acute", "lung", "lung_big_model", "aneurysms", "heart"]:  # , "tbad_dataset"]:
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


def save_predict(data_batch, pred, path: Path):

    path = Path(path)
    sample_name = Path(data_batch['label_meta_dict']['filename_or_obj'][0]).stem.split('.')[0]

    img = np.array(data_batch["image"][0].cpu())
    lab = np.array(data_batch["label"][0].cpu())
    out = np.array(pred[0].cpu())

    nii = nib.Nifti1Image(lab[0], affine=np.array(data_batch['label_meta_dict']['affine'][0].cpu()))
    nib.save(nii, path / f'lab_{sample_name}_{np.sum(lab)}.nii.gz') 
    
    # if lab.sum() > 0:
    #     lab_c = get_center_mask(lab)
    #     out_c = get_center_mask(out)
    #     dist = round(np.abs(lab_c - out_c).mean(), 3) 
    # else:
    #     dist = -1

    nii = nib.Nifti1Image(out[0], affine=np.array(data_batch['label_meta_dict']['affine'][0].cpu()))
    nib.save(nii, path / f'pred_{sample_name}_{np.sum(out)}.nii.gz')   

    # nib.save(nii, path / f'lab_{sample_name}_{np.sum(lab)}_{np.sum(out)}_metr{round(1. - float(np.sum(np.abs(lab - out)) / (max(np.sum(lab), np.sum(out))) + 1e-6), 3)}_dist{dist}.nii.gz')

    nii = nib.Nifti1Image(img[0], affine=np.array(data_batch['image_meta_dict']['affine'][0].cpu()))
    nib.save(nii, path / f'img_{sample_name}.nii.gz')



@torch.no_grad()
def val_and_save_one_epoch(
    model: torch.nn.Module,
    data_flag: str,
    inference: monai.inferers.Inferer,
    data_loader: torch.utils.data.DataLoader,
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
    for data_batch in tqdm(data_loader):
        logits = inference(data_batch["image"].to(device), model)
        val_outputs = [post_trans(i) for i in logits]
 
        save_predict(data_batch, val_outputs, path)
        
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=data_batch["label"].to(device))
  
    _, metrics_dict = calc_metrics_dict(
        metrics, accelerator, data_flag, is_train=False
    )

    accelerator.print(
        f"Epoch [{epoch}/{config.trainer.num_epochs}] metric {metrics_dict}"
    )

    return metrics_dict


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


def get_sampels_name(data_loader):
    
    sampels = []
    for data_batch in tqdm(data_loader):
        print(data_batch.keys())
        sampels.extend(
            Path(sample_name).stem.split('.')[0]
            for sample_name in data_batch['label_meta_dict']['filename_or_obj']
        )
        
    return sampels


if __name__ == "__main__":

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    config, data_flag, is_HepaticVessel = load_config(
        config_filename="config.yml", mode="r"
    )
    print(config)

    config.trainer.start_unlab_epoch = int(config.trainer.start_unlab_epoch_ratio * config.trainer.num_epochs)

    same_seeds(config.trainer.seed)
    logging_dir = get_new_experiment_dir(config, data_flag)

    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    Logger(logging_dir)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    log_metric_dict = dict()

    ds_name = config.data_root.split('/')[-1]

    path = Path(f'./eval/{ds_name}/save_unet_aniv_split')
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.json").open("w") as fp:
        json.dump(config , fp)
    

    accelerator.print("Load Model...")
    model = create_model(config, n_filters=8)

    accelerator.print("Load Dataloader...")
    config_copy = copy.copy(config)

    train_loader, val_loader, unlab_loader = get_dataloader(config, data_flag, needs_unlab=False)

    print(train_loader.dataset)

    # val_names   = get_sampels_name(val_loader)
    # train_names = get_sampels_name(train_loader)

    loader, data_list = load_sampels(config)
    # loader_names = get_sampels_name(loader)

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

    base_exp_path_save = get_new_experiment_dir(config, data_flag, root="model_store")

    print(list(base_exp_path_save.iterdir()))

    exit()

    list_epochs = list(base_exp_path_save.iterdir())[1:]

    print('Start eval')
    print(sorted(list_epochs, key=lambda x: int(x.name.split('_')[-1]))[10:])


    for epoch_folder in sorted(list_epochs, key=lambda x: int(x.name.split('_')[-1]))[10:]:

        epoch = int(epoch_folder.name.split('_')[-1])

        model = utils.load_pretrain_model(
            str(epoch_folder / "pytorch_model.bin"),
            model,
            accelerator,
        )

        device = next(model.parameters()).device

        # val
        metrics_dict = val_and_save_one_epoch(
            model,
            data_flag,
            inference,
            loader,
            metrics,
            post_trans,
            accelerator,
            epoch=epoch,
            path=path / f'epoch_{epoch}',
            device=device
        )

        log_metric_dict.update(
            {epoch: metrics_dict}
        )


        # unlab_list = [] 
        # for data_batch in tqdm(unlab_loader):
        #     unlab_list.extend(data_batch['label_meta_dict']['filename_or_obj'])
        
        # pprint(unlab_list)

    with (path / "metrics.json").open("w") as fp:
        json.dump(log_metric_dict , fp)
