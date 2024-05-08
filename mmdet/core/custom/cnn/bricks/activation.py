# import torch
# import torch.nn as nn
#
# from mmcv.utils import build_from_cfg
# from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS
#
# for module in [
#         nn.SiLU
# ]:
#     ACTIVATION_LAYERS.register_module(module=module)
#
#
# def build_activation_layer(cfg):
#     """Build activation layer.
#
#     Args:
#         cfg (dict): The activation layer config, which should contain:
#             - type (str): Layer type.
#             - layer args: Args needed to instantiate an activation layer.
#
#     Returns:
#         nn.Module: Created activation layer.
#     """
#     return build_from_cfg(cfg, ACTIVATION_LAYERS)
