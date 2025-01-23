from monai import losses


def get_loss_functions(trainer_config):

    loss_functions = {
        "focal_loss": (
            losses.FocalLoss(
                to_onehot_y=False,
                # weight=trainer_config.focal_class_weights,
                gamma=trainer_config.gamma,
            ),
            trainer_config.focal_loss_ratio,
        ),
        # "dice_loss": (
        #     losses.DiceLoss(
        #         to_onehot_y=False,
        #         sigmoid=True,
        #         smooth_nr=1e-5,
        #         smooth_dr=1e-5,
        #     ),
        #     trainer_config.dice_loss_ratio,
        # ),
        # "gen_dice_loss": (
        #     losses.GeneralizedDiceLoss(
        #         to_onehot_y=False,
        #         sigmoid=True,
        #         smooth_nr=1e-5,
        #         smooth_dr=1e-5,
        #     ),
        #     trainer_config.dice_loss_ratio,
        # ),
        "tversky_loss": (
            losses.TverskyLoss(
            to_onehot_y=False, sigmoid=True, alpha=0.4, beta=0.7
            ),
            trainer_config.dice_loss_ratio,
        )
    }

    return loss_functions


def calc_total_loss(logits, label, loss_functions, accelerator = None, step = 0, train=True):
    log = ""
    total_loss = 0
    name_stage = "Train" if train else "Val"
    for name in loss_functions:
        loss_fn, ratio = loss_functions[name]
        loss = ratio * loss_fn(logits, label)
        if accelerator is not None:
            accelerator.log({f"{name_stage}/" + name: float(loss)}, step=step)
        log += f" {name} {float(loss):1.5f} "
        total_loss += loss

    return total_loss, log