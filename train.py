import os
import sys
import pymesh
import torch

from data.human_data import SMPL_DATA
from data.animal_data import SMAL_DATA
from ver2ver_trainer import Ver2VerTrainer
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.util import print_current_errors


# parse options
opt = TrainOptions().parse()
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
if opt.dataset_mode == 'human':
    dataset = SMPL_DATA(opt, True)
    if opt.use_unlabelled:
        dataset_unlabelled = SMPL_DATA(opt, True, False)
elif opt.dataset_mode == 'animal':
    dataset = SMAL_DATA(opt, True)
    if opt.use_unlabelled:
        dataset_unlabelled = SMAL_DATA(opt, True, False)
else:
    raise ValueError("|dataset_mode| is invalid")

if opt.percentage > 0:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.nThreads), drop_last=opt.isTrain)
    iter_data = iter(dataloader)
if opt.use_unlabelled:
    dataloader_unlabelled = torch.utils.data.DataLoader(dataset_unlabelled, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.nThreads), drop_last=opt.isTrain)
    iter_data_unlabelled = iter(dataloader_unlabelled)

# create tool for counting iterations
iter_per_epoch = opt.num_total // opt.batchSize
iter_counter = IterationCounter(opt, iter_per_epoch * opt.batchSize)

# create trainer for our model
trainer = Ver2VerTrainer(opt, iter_counter=iter_counter)

# save root of the optputs
save_root = os.path.join(opt.exp_dir, 'training_output')
if not os.path.exists(save_root):
    os.makedirs(save_root)


for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i in range(iter_counter.epoch_iter, iter_per_epoch):
        iter_counter.record_one_iteration()

        # labelled flag
        if opt.use_unlabelled and ((i % 2) == 1 or opt.percentage == 0):
            iter_is_labelled = False
        else:
            iter_is_labelled = True

        if (not iter_is_labelled) and (opt.percentage > 0) and (i % 6) != 5:
            cross_iter = True
            id_from_labelled = True if (i % 6) == 1 else False
        else:
            cross_iter = False
            id_from_labelled = False

        # get data
        if iter_is_labelled or cross_iter:
            try:
                data_i = next(iter_data)
            except:
                iter_data = iter(dataloader)
                data_i = next(iter_data)
        if not iter_is_labelled:
            try:
                data_unlabelled_i = next(iter_data_unlabelled)
            except:
                iter_data_unlabelled = iter(dataloader_unlabelled)
                data_unlabelled_i = next(iter_data_unlabelled)

        gt_points = None
        if iter_is_labelled: # labelled
            identity_points, pose_points, gt_points, id_face, pose_face = data_i[:5]
            pose_points2 = data_i[5] if opt.use_unlabelled else None
        elif not cross_iter: # unlabelled
            identity_points, pose_points, _, id_face, pose_face = data_unlabelled_i[:5]
            pose_points2 = data_unlabelled_i[5] if opt.use_unlabelled else None
        elif id_from_labelled: # cross, id from labelled
            identity_points, _, _, id_face, _ = data_i[:5]
            _, pose_points, _, _, pose_face = data_unlabelled_i[:5]
            pose_points2 = data_unlabelled_i[5] if opt.use_unlabelled else None
        else: # cross, id from unlabelled
            _, pose_points, _, _, pose_face = data_i[:5]
            identity_points, _, _, id_face, _ = data_unlabelled_i[:5]
            pose_points2 = data_i[5] if opt.use_unlabelled else None

        # training
        trainer.train_model(identity_points, pose_points, gt_points, id_face, iter_is_labelled, pose_face, pose_points2)

        # print loss
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            try:
                print_current_errors(opt, epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            except OSError as err:
                print(err)

        # save mesh
        if opt.save_training_output and iter_counter.needs_displaying():
            try:
                pymesh.save_mesh_raw(save_root + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_id.obj',
                                                identity_points[0,:,:3].cpu().numpy(),id_face[0,:,:].cpu().numpy())
                pymesh.save_mesh_raw(save_root + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_pose.obj',
                                                pose_points[0,:,:].cpu().numpy(),pose_face[0,:,:].cpu().numpy())
                pymesh.save_mesh_raw(save_root + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_warp.obj',
                                                trainer.out['warp_out'][0,:,:].cpu().detach().numpy(),id_face[0,:,:].cpu().numpy())
                pymesh.save_mesh_raw(save_root + '/' + str(epoch) + '_' + str(iter_counter.total_steps_so_far) + '_out.obj',
                                                trainer.get_latest_generated().data[0,:,:].cpu().detach().numpy().transpose(1,0),id_face[0,:,:].cpu().numpy())
            except OSError as err:
                print(err)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                (epoch, iter_counter.total_steps_so_far))
            try:
                trainer.save('latest')
                iter_counter.record_current_iter()
            except OSError as err:
                print(err)

    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        try:
            trainer.save('latest')
            trainer.save(epoch)
        except OSError as err:
            print(err)

    trainer.update_learning_rate(epoch)

print('Training was successfully finished.')
