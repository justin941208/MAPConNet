import os
import torch.utils.data as data
import torch
import numpy as np
import pymesh
from data.data_utils import data_list, face_reverse

class SMPL_DATA(data.Dataset):
    def __init__(self, opt, train, labelled=True):
        self.opt = opt
        self.train = train
        self.labelled = labelled
        self.vertex_num = 6890
        self.path = opt.dataroot
        self.datapath = []
        self.num_total = opt.num_total

        if self.train:
            if self.labelled:
                datapath_file = None if opt.percentage == 0 else f'datapath_labelled_{opt.percentage}%.txt'
            else:
                datapath_file = None if opt.percentage == 100 else f'datapath_unlabelled_{100 - opt.percentage}%.txt'

            if datapath_file is not None:
                datapath_file = os.path.join('data', 'datalists', datapath_file)
                if not os.path.exists(datapath_file):
                    data_list(opt.percentage, self.num_total, 'human')

                with open(datapath_file, 'r') as f:
                    for l, line in enumerate(f):
                        if l == 0:
                            self.ids = [int(i) for i in line.split(',')]
                            self.num_id = len(self.ids)
                        elif l == 1:
                            self.poses = [int(p) for p in line.split(',')]
                            self.num_pose = len(self.poses)
                        else:
                            identity_i = int(line.split(',')[0])
                            identity_p = int(line.split(',')[1])
                            self.datapath.append([identity_i, identity_p])
            else:
                self.num_id = self.num_pose = 0
            print(f'Training set. Human. Labelled: {self.labelled}. Size: {len(self.datapath)}. #IDs: {self.num_id}. #Poses: {self.num_pose}.')
        else:
            self.lines = [line for line in open(os.path.join('data', 'datalists', 'human_test_list'), "r")]
            print(f'Test set. Human. Size: {len(self.lines)}.')

    def __getitem__(self, index):
        if self.train:
            np.random.seed()
            mesh_set = self.datapath[index]
            identity_mesh_i = mesh_set[0]
            identity_mesh_p = mesh_set[1]
            pose_mesh_i = np.random.choice(self.ids, replace=True)
            pose_mesh_p = np.random.choice(self.poses, replace=True)
            identity_mesh = pymesh.load_mesh(self.path+'id'+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'.obj')
            pose_mesh = pymesh.load_mesh(self.path+'id'+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'.obj')
            gt_mesh = pymesh.load_mesh(self.path+'id'+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'.obj')
            gt_mesh_name = 'id'+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'.obj'
            if self.opt.use_unlabelled:
                ps = np.random.choice(self.poses, size=2, replace=False)
                pose_mesh2_p = ps[0] if ps[0] != pose_mesh_p else ps[1]
                pose_mesh2 = pymesh.load_mesh(self.path+'id'+str(pose_mesh_i)+'_'+str(pose_mesh2_p)+'.obj')
        else:
            data_list = self.lines[index].strip('\n').split(' ')
            id_mesh_name = data_list[0]
            pose_mesh_name = data_list[1]
            gt_mesh_name = data_list[2]

            identity_mesh = pymesh.load_mesh(self.path + id_mesh_name)
            pose_mesh = pymesh.load_mesh(self.path + pose_mesh_name)
            gt_mesh = pymesh.load_mesh(self.path + gt_mesh_name)

        identity_points = identity_mesh.vertices
        identity_faces = identity_mesh.faces
        pose_points = pose_mesh.vertices
        pose_faces = pose_mesh.faces
        gt_points = gt_mesh.vertices
        if self.train and self.opt.use_unlabelled:
            pose_points2 = pose_mesh2.vertices

        # pose points
        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))
        if self.train and self.opt.use_unlabelled:
            pose_points2 = pose_points2 - (pose_mesh2.bbox[0] + pose_mesh2.bbox[1]) / 2
            pose_points2 = torch.from_numpy(pose_points2.astype(np.float32))

        # identity points
        identity_points = identity_points - (identity_mesh.bbox[0] + identity_mesh.bbox[1]) / 2
        identity_points = torch.from_numpy(identity_points.astype(np.float32))

        # ground truth points
        gt_points = gt_points - (gt_mesh.bbox[0] + gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        new_id_faces = identity_faces
        new_pose_faces = pose_faces

        random_sample = random_sample2 = None

        # Before input, shuffle the vertices randomly to be close to real-world problems.
        random_sample = np.random.choice(self.vertex_num,size=self.vertex_num,replace=False)
        random_sample2 = np.random.choice(self.vertex_num,size=self.vertex_num,replace=False)
        pose_points = pose_points[random_sample2]
        identity_points = identity_points[random_sample]
        gt_points = gt_points[random_sample]
        if self.train and self.opt.use_unlabelled:
            pose_points2 = pose_points2[random_sample2]

        new_id_faces = face_reverse(identity_faces, random_sample)
        new_pose_faces = face_reverse(pose_faces, random_sample2)

        if self.opt.isTrain:
            if not self.opt.use_unlabelled:
                return identity_points, pose_points, gt_points, new_id_faces, new_pose_faces
            return identity_points, pose_points, gt_points, new_id_faces, new_pose_faces, pose_points2
        elif not (self.train and self.opt.use_unlabelled):
            return identity_points, pose_points, (gt_points, gt_mesh_name), new_id_faces, new_pose_faces
        return identity_points, pose_points, (gt_points, gt_mesh_name), new_id_faces, new_pose_faces, pose_points2

    def __len__(self):
        if self.train:
            return len(self.datapath)
        return len(self.lines)
