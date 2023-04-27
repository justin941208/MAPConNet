import os
import torch

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.ver2ver_model import Ver2VerModel

class Ver2VerTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, iter_counter=None):
        self.opt = opt
        self.ver2ver_model = Ver2VerModel(opt)
        if len(opt.gpu_ids) > 1:
            self.ver2ver_model = DataParallelWithCallback(self.ver2ver_model,
                                                          device_ids=opt.gpu_ids)
            self.ver2ver_model_on_one_gpu = self.ver2ver_model.module
        elif len(opt.gpu_ids) == 1:
            self.ver2ver_model.to(opt.gpu_ids[0])
            self.ver2ver_model_on_one_gpu = self.ver2ver_model
        else:
            self.ver2ver_model_on_one_gpu = self.ver2ver_model

        if opt.isTrain:
            self.optimizer = self.ver2ver_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
            if opt.continue_train and opt.which_epoch == 'latest':
                checkpoint = torch.load(os.path.join(opt.checkpoints_dir, 'optimizer.pth'))
                self.optimizer.load_state_dict(checkpoint['G'])
                if opt.lr_override is None:
                    self.old_lr = checkpoint['lr']
                else:
                    self.old_lr = opt.lr_override
                self.update_learning_rate(iter_counter.first_epoch - 1)

    def train_model(self, identity_points, pose_points, gt_points, id_face, iter_is_labelled, pose_face, pose_points2):
        self.optimizer.zero_grad()
        losses, out = self.ver2ver_model(identity_points, pose_points, gt_points, id_face, mode='train',
                                         iter_is_labelled=iter_is_labelled, pose_face=pose_face, pose_points2=pose_points2)
        loss = sum(losses.values()).mean()
        loss.backward()
        self.optimizer.step()
        self.losses = {}
        for k in losses.keys():
            self.losses[k] = losses[k].detach().cpu()
        self.losses['lr'] = self.optimizer.param_groups[0]['lr']
        self.out = {}
        for k in out.keys():
            self.out[k] = out[k].detach().cpu()

    def get_latest_losses(self):
        return {**self.losses}

    def get_latest_generated(self):
        return self.out['fake_points']

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch, _use_new_zipfile_serialization=False):
        self.ver2ver_model_on_one_gpu.save(epoch)
        if epoch == 'latest':
            torch.save({'G': self.optimizer.state_dict(),
                        'lr':  self.old_lr},
                        os.path.join(self.opt.checkpoints_dir, 'optimizer.pth'),
                        _use_new_zipfile_serialization=_use_new_zipfile_serialization)
                        # }, os.path.join(self.opt.checkpoints_dir, self.opt.dataset_mode, 'optimizer.pth'))

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            new_lr_G = new_lr

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
