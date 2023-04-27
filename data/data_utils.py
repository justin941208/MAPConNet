import os
import numpy as np

def face_reverse(faces, random_sample):
    face_dict = {}
    for i in range(len(random_sample)):
        face_dict[random_sample[i]] = i
    new_f = []
    for i in range(len(faces)):
        new_f.append([face_dict[faces[i][0]],face_dict[faces[i][1]],face_dict[faces[i][2]]])
    new_face = np.array(new_f)
    return new_face

def data_list(percentage, num_total, mode):
    datapath_labelled = []
    datapath_unlabelled = []
    if mode == 'human':
        ids_all = np.arange(0, 16)
        poses_all = np.arange(200, 600)
    elif mode == 'animal':
        ids_all = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35, 38, 39])
        poses_all = np.arange(0, 400)
    else:
        raise Exception

    np.random.seed(1234)
    num_id = round(len(ids_all) * percentage / 100)
    num_pose = round(len(poses_all) * percentage / 100)
    ids = np.random.choice(ids_all, size=num_id, replace=False)
    poses = np.random.choice(poses_all, size=num_pose, replace=False)
    unused_ids = [i for i in ids_all if i not in ids]
    unused_poses = [i for i in poses_all if i not in poses]

    if len(ids) > 0 and len(poses) > 0:
        for _ in range(num_total):
            identity_i = np.random.choice(ids, replace=True)
            identity_p = np.random.choice(poses, replace=True)
            datapath_labelled.append([identity_i, identity_p])
        fname = f'datapath_labelled_{percentage}%.txt'
        if mode == 'animal':
            fname = 'animal_' + fname
        with open(os.path.join('data', 'datalists', fname), 'w') as f:
            f.write(','.join([str(i) for i in ids]) + '\n')
            f.write(','.join([str(p) for p in poses]) + '\n')
            for i, p in datapath_labelled:
                f.write(f'{i},{p}\n')

    if len(unused_ids) > 0 and len(unused_poses) > 0:
        for _ in range(num_total):
            identity_i = np.random.choice(unused_ids, replace=True)
            identity_p = np.random.choice(unused_poses, replace=True)
            datapath_unlabelled.append([identity_i, identity_p])
        fname = f'datapath_unlabelled_{100 - percentage}%.txt'
        if mode == 'animal':
            fname = 'animal_' + fname
        with open(os.path.join('data', 'datalists', fname), 'w') as f:
            f.write(','.join([str(i) for i in unused_ids]) + '\n')
            f.write(','.join([str(p) for p in unused_poses]) + '\n')
            for i, p in datapath_unlabelled:
                f.write(f'{i},{p}\n')
