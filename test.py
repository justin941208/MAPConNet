import os
import torch
import numpy as np
import pymesh
from options.test_options import TestOptions
from models.ver2ver_model import Ver2VerModel

from data.human_data import SMPL_DATA
from data.animal_data import SMAL_DATA
from tqdm import tqdm


def format_outputs(out):
    to_transpose = ['identity_points', 'pose_points', 'id_features', 'pose_features', 'warp_out']
    for key in out.keys():
        if key in to_transpose:
            out[key] = out[key].transpose(2,1)
        if key not in ['id_face', 'pose_face']:
            out[key] = out[key].squeeze().detach().cpu().numpy()
    return out

def test(opt, dataset):
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    model = Ver2VerModel(opt, opt.test_epoch)
    model.eval()

    metric_test = 0.0
    for i in tqdm(range(len(dataset)), desc=f'Testing epoch {opt.test_epoch}'):
        mesh_num = i + 1
        identity_points, pose_points, gt_points_and_name, id_face, pose_face = dataset[i]
        gt_points, gt_mesh_name = gt_points_and_name

        # generate results
        out = format_outputs(model(identity_points, pose_points, gt_points, None, mode='inference'))

        # calculate results
        metric_test += float(out[opt.metric])

        # save the generated meshes
        # if opt.save_output:
        save_root = os.path.join(opt.results_dir, opt.test_epoch, 'outputs')
        os.makedirs(save_root, exist_ok=True)
        if opt.dataset_mode == 'human':
            pymesh.save_mesh_raw(save_root + '/' + gt_mesh_name, out['fake_points'], id_face)
        elif opt.dataset_mode == 'animal':
            pymesh.save_mesh_raw(save_root + '/' + gt_mesh_name.strip('.ply') + '.obj', out['fake_points'], id_face)
        np.save(os.path.join(save_root, gt_mesh_name[:-4] + '_warp'), out['warp_out'])

    numbers_dir = os.path.join(opt.results_dir, opt.test_epoch)
    os.makedirs(numbers_dir, exist_ok=True)
    print(f'Final {opt.metric} for ' + opt.dataset_mode + ' is ' + str(metric_test/mesh_num))
    with open(os.path.join(numbers_dir, f'{opt.metric}_{opt.test_epoch}.txt'), 'w') as f:
        f.write(f'{opt.metric}: {metric_test/mesh_num}')

if __name__ == '__main__':
    opt = TestOptions().parse()

    if opt.dataset_mode == 'human':
        print('test model on unseen data in SMPL')
        dataset = SMPL_DATA(opt, False)
    elif opt.dataset_mode == 'animal':
        print('test model on unseen data in SMAL')
        dataset = SMAL_DATA(opt, False)
    else:
        raise ValueError("|dataset_mode| is invalid")

    with torch.no_grad():
        test(opt, dataset)
