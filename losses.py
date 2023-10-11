import torch


def l1(output, target):
    return torch.mean(torch.abs(output - target))


def l1_wav(output_dict, target_dict):
	return l1(output_dict['segment'], target_dict['segment'])


def get_loss_function(loss_type):
    if loss_type == "l1_wav":
        return l1_wav

    else:
        raise NotImplementedError("Error!")
