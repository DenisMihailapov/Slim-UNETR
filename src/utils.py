import os
import yaml
from pathlib import Path
import sys
from easydict import EasyDict
from collections import OrderedDict


import numpy as np
import torch
from accelerate import Accelerator
from timm.models.layers import trunc_normal_
from torch import nn


def load_config(config_filename="config.yml", mode="r"):
    # load yml
    config = None
    with open(config_filename, mode, encoding="utf-8") as conf_file:
        config = EasyDict(yaml.load(conf_file, Loader=yaml.FullLoader))

    if config.is_brats2021:
        config = config.brats2021
        data_flag = "brats2021"
        is_HepaticVessel = False
    elif config.is_brain2019:
        config = config.brain2019
        data_flag = "brain2019"
        is_HepaticVessel = False
    elif config.is_brain2019_small:
        config = config.brain2019_small
        data_flag = "brain2019_small"
        is_HepaticVessel = False
    elif config.is_hepatic_vessel2021:
        config = config.hepatic_vessel2021
        data_flag = "hepatic_vessel2021"
        is_HepaticVessel = True
    elif config.is_heart:
        config = config.heart
        data_flag = "heart"
        is_HepaticVessel = False
    elif config.is_acute:
        config = config.acute
        data_flag = "acute"
        is_HepaticVessel = False
    elif config.is_lung:
        config = config.lung
        data_flag = "lung"
        is_HepaticVessel = False
    elif config.is_lung_big_model:
        config = config.lung_big_model
        data_flag = "lung_big_model"
        is_HepaticVessel = False
    elif config.is_tbad_dataset:
        config = config.tbad_dataset
        data_flag = "tbad_dataset"
        is_HepaticVessel = False
    else:
        raise ValueError("Please set dataset in config file")

    return config, data_flag, is_HepaticVessel


def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith("http"):
        state_dict = torch.hub.load_state_dict_from_url(
            download_path,
            model_dir=save_path,
            check_hash=check_hash,
            map_location=torch.device("cpu"),
        )
    else:
        state_dict = torch.load(download_path, map_location=torch.device("cpu"))
    return state_dict


def resume_train_state(
    model,
    path: str,
    train_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    epoch: int = -1,
):
    try:
        # Get the most recent checkpoint
        base_path = Path(os.getcwd()) / "model_store"
        base_path /= path
        print("base_path:", base_path)

        dirs = [f for f in base_path.glob("*_*") if f.is_dir()]
        dirs.sort(
            key=lambda f: int(f.name.split("_")[-1])
        )  # Sorts folders by date modified, most recent checkpoint is the last

        if epoch != -1:
            ind_epoch = 0
            found = False
            for f in dirs:
                if int(f.name.split("_")[-1]) == epoch:
                    found = True
                    break
                ind_epoch += 1

            if found:
                epoch = ind_epoch

        accelerator.print(f"try to load {str(dirs[epoch])} train stage")
        model = load_pretrain_model(
            str(dirs[epoch] / "pytorch_model.bin"), model, accelerator
        )
        starting_epoch = int(dirs[epoch].name.replace("epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        accelerator.print(
            f"Load state training success ！Start from {starting_epoch} epoch"
        )
        return model, starting_epoch, step, step
    except Exception as e:
        accelerator.print(e)
        accelerator.print("Load training status failed ！")
        return model, 0, 0, 0


def load_pretrain_model(pretrain_path: str, model: nn.Module, accelerator: Accelerator):
    try:
        state_dict = load_model_dict(pretrain_path)
        state_dict = {k[0].lower() + k[1:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        accelerator.print("Successfully loaded the training model！")
        return model
    except Exception as e:
        accelerator.print(e)
        accelerator.print("Failed to load the training model！")
        return model


def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Logger(object):
    def __init__(self, logdir: Path):
        self.console = sys.stdout
        self.log_file = None

        if logdir is not None:
            logdir.mkdir(exist_ok=True, parents=True)
            self.log_file = (logdir / "log.txt").open("w", encoding="utf-8")

        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()
