from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class PartSepHook(Hook):
    def __init__(self,
                 num_last_epochs=1):
        self.num_last_epochs = num_last_epochs

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) >= runner.max_epochs - self.num_last_epochs:
            runner.logger.info('interpolate_epoch')
            model._modules['bbox_head'].train_sep = False