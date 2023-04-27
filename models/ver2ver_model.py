import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util


class Ver2VerModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt, which_epoch=None):
        super().__init__()

        self.opt = opt
        self.net = torch.nn.ModuleDict(self.initialize_networks(opt, which_epoch))

    def forward(self, identity_points, pose_points, gt_points, id_face, mode,
                iter_is_labelled=True, pose_face=None, pose_points2=None):
        self.iter_is_labelled = iter_is_labelled
        if mode == 'inference':
            identity_points = identity_points.transpose(1,0).unsqueeze(0)
            pose_points = pose_points.transpose(1,0).unsqueeze(0)
            if len(self.opt.gpu_ids) >= 1:
                identity_points = identity_points.cuda()
                pose_points = pose_points.cuda()
                gt_points = gt_points.cuda()
        else:
            identity_points=identity_points.transpose(2,1) #(bs, 3, n)
            pose_points=pose_points.transpose(2,1)
            if len(self.opt.gpu_ids) >= 1:
                identity_points=identity_points.cuda()
                pose_points=pose_points.cuda()

            if self.opt.use_unlabelled:
                pose_points2=pose_points2.transpose(2,1)
                if len(self.opt.gpu_ids) >= 1:
                    pose_points2=pose_points2.cuda()

            if self.iter_is_labelled:
                gt_points=gt_points.transpose(2,1)
                if len(self.opt.gpu_ids) >= 1:
                    gt_points=gt_points.cuda()

        generated_out = {}
        if mode == 'train':

            out = {}
            if self.iter_is_labelled:
                loss, generated_out = self.compute_loss(identity_points, pose_points, gt_points, id_face,
                                                        return_Tm=True, return_features=True)
                self.get_output_features(generated_out)
                loss.update(self.mesh_loss(generated_out, reorder=True, sc=True))
                loss.update(self.point_loss(generated_out))

                if self.opt.save_training_output:
                    out['fake_points'] = generated_out['fake_points'].detach().cpu()
            else:
                loss = {}

                ### Cross-consistency
                loss_cc, generated_out_cc = self.compute_loss(pose_points2, pose_points, pose_points, pose_face, return_features=True)
                self.get_output_features(generated_out_cc)
                loss_cc.update(self.mesh_loss(generated_out_cc, reorder=False))
                loss_cc.update(self.point_loss(generated_out_cc))

                ### Self-consistency
                # first pass
                id_features = self.net['netCorr'](identity_points, None, encode=True)
                loss_fp, first_pass = self.compute_loss(identity_points, pose_points, None, id_face,
                                                        id_features=id_features, pose_features=generated_out_cc['pose_features'],
                                                        return_Tm=True, return_features=True, detach=True, skip_rec=True, skip_edge=True)
                self.get_output_features(first_pass)
                loss_fp.update(self.mesh_loss(first_pass, reorder=True, sc=True))
                loss_fp.update(self.point_loss(first_pass))
                first_pass['fake_features_detached'] = self.net['netCorr'](first_pass['fake_points'].detach(), None, encode=True)

                # second pass
                loss_sc, generated_out_sc = self.compute_loss(pose_points2, first_pass['fake_points'].detach(), pose_points, pose_face,
                                                              id_features=generated_out_cc['id_features'], pose_features=first_pass['fake_features_detached'],
                                                              return_Tm=True, return_features=True)
                self.get_output_features(generated_out_sc)
                loss_sc.update(self.mesh_loss(generated_out_sc, reorder=True, sc=True))
                loss_sc.update(self.point_loss(generated_out_sc))

                # summarising all loss terms
                for k in loss_cc.keys():
                    loss[k + '_cc'] = loss_cc[k]
                for k in loss_sc.keys():
                    loss[k + '_sc'] = loss_sc[k]
                for k in loss_fp.keys():
                    loss[k + '_fp'] = loss_fp[k]

                if self.opt.save_training_output:
                    out['fake_points_cc'] = generated_out_cc['fake_points'].detach().cpu()
                    out['fake_points_first_pass'] = first_pass['fake_points'].detach().cpu()
                    out['fake_points_sc'] = generated_out_sc['fake_points'].detach().cpu()

            if self.opt.save_training_output:
                out['identity_points'] = identity_points.detach().cpu()
                out['pose_points'] = pose_points.detach().cpu()
            return loss, out

        elif mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(identity_points, pose_points, gt_points)

            out['identity_points'] = identity_points
            out['pose_points'] = pose_points
            out['id_face'] = id_face
            out['pose_face'] = pose_face
            return out
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list()
        G_params += [{'params': self.net['netG'].parameters(), 'lr': opt.lr}]
        G_params += [{'params': self.net['netCorr'].parameters(), 'lr': opt.lr}]

        beta1, beta2 = opt.beta1, opt.beta2
        G_lr = opt.lr

        optimizer = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)

        return optimizer

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netCorr'], 'Corr', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt, which_epoch):
        net = {}
        net['netG'] = networks.define_G(opt)
        net['netCorr'] = networks.define_Corr(opt)
        if opt.isTrain:
            print(net['netCorr'])
            print(net['netG'])

        if not opt.isTrain or opt.continue_train:
            if which_epoch is None:
                which_epoch = opt.which_epoch
            net['netG'] = util.load_network(net['netG'], 'G', which_epoch, opt)
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', which_epoch, opt)

        return net

    def compute_loss(self, identity_points, pose_points, gt_points, id_face, id_rec=False,
                     return_Tm=False, return_features=False, id_features=None, pose_features=None, detach=False, skip_rec=False, skip_edge=False):
        losses = {}

        if (id_features is None) or (pose_features is None):
            generate_out = self.generate_fake(identity_points, pose_points, detach=detach, return_Tm=return_Tm, return_features=return_features)
        else:
            generate_out = self.generate_fake(None, pose_points, detach=detach, return_Tm=return_Tm, return_features=return_features,
                                              id_features=id_features, pose_features=pose_features)

        # edge loss
        if not skip_edge:
            losses['edge_loss'] = 0.0
            if id_rec:
                identity_points = gt_points
            for i in range(len(identity_points)):
                f = id_face[i].cpu().numpy()
                v = identity_points[i].detach().transpose(0,1).cpu().numpy()
                losses['edge_loss'] = losses['edge_loss'] + util.compute_score(generate_out['fake_points'][i].transpose(1,0).unsqueeze(0),f,util.get_target(v,f,1))
            losses['edge_loss'] = losses['edge_loss']/len(identity_points) * self.opt.lambda_edge

        # reconstruction loss
        if not skip_rec:
            losses['rec_loss'] = torch.mean((generate_out['fake_points'] - gt_points)**2) * self.opt.lambda_rec

        return losses, generate_out

    def generate_fake(self, identity_points, pose_points, detach=False,
                      return_Tm=False, return_features=False, id_features=None, pose_features=None):
        generate_out = {}

        # correspondence
        if (id_features is None) or (pose_features is None):
            corr_out = self.net['netCorr'](pose_points, identity_points, return_Tm=return_Tm)
        else:
            corr_out = self.net['netCorr'](pose_points, None, decode=True, return_Tm=return_Tm, id_features=id_features, pose_features=pose_features)

        if return_features:
            generate_out.update(corr_out)

        # mesh refinement
        id_features_in = corr_out['id_features']

        if detach:
            generate_out['fake_points'] = self.net['netG'](id_features_in, corr_out['warp_out']).detach()
        else:
            generate_out['fake_points'] = self.net['netG'](id_features_in, corr_out['warp_out'])

        return generate_out

    def inference(self, identity_points, pose_points, gt_points):
        generate_out = {}

        corr_out = self.net['netCorr'](pose_points, identity_points)
        generate_out['fake_points'] = self.net['netG'](corr_out['id_features'], corr_out['warp_out'])

        generate_out = {**generate_out, **corr_out}

        # evaluation
        generate_out['warp_out'] = generate_out['warp_out'].transpose(2, 1)
        key = 'fake_points'
        bbox = torch.tensor([[torch.max(generate_out[key][:,0]), torch.max(generate_out[key][:,1]), torch.max(generate_out[key][:,2])],
                            [torch.min(generate_out[key][:,0]), torch.min(generate_out[key][:,1]), torch.min(generate_out[key][:,2])]], device=generate_out[key].device)
        generate_out[key] = generate_out[key].squeeze().transpose(1,0) - (bbox[0] + bbox[1] ) / 2

        if self.opt.metric == 'PMD':
            generate_out['PMD'] = torch.mean((generate_out[key] - gt_points)**2)
        elif self.opt.metric == 'CD':
            generate_out['CD'] = self.chamfer_distance(generate_out[key], gt_points, 'mean')

        return generate_out

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def get_output_features(self, generated_out):
        generated_out['out_features'] = self.net['netCorr'](generated_out['warp_out'].transpose(2, 1), None, encode=True)

    def mesh_loss(self, generated_out, reorder, sc=False):
        pose_feat = generated_out['pose_features']
        id_feat = generated_out['id_features']
        out_feat = generated_out['out_features']
        if reorder:
            pose_feat = torch.matmul(generated_out['T_m_bin'].detach(), pose_feat.permute(0, 2, 1)).transpose(2, 1)

        loss = {}
        id_dim = self.opt.id_dim
        if sc:
            mesh_loss_id = self.triplet_loss(out_feat[:, :id_dim, :], id_feat[:, :id_dim, :], pose_feat[:, :id_dim, :], meshwise=True, margin=self.opt.margin_mesh)
            mesh_loss_pose = self.triplet_loss(out_feat[:, id_dim:, :], pose_feat[:, id_dim:, :], id_feat[:, id_dim:, :], meshwise=True, margin=self.opt.margin_mesh)
        else:
            mesh_loss_id = self.triplet_loss(pose_feat[:, :id_dim, :], id_feat[:, :id_dim, :], out_feat[:, :id_dim, :], meshwise=True, margin=self.opt.margin_mesh)
            mesh_loss_pose = self.triplet_loss(pose_feat[:, id_dim:, :], out_feat[:, id_dim:, :], id_feat[:, id_dim:, :], meshwise=True, margin=self.opt.margin_mesh)
        loss['mesh_loss'] = (mesh_loss_id + mesh_loss_pose) * self.opt.lambda_mesh
        return loss

    def point_loss(self, generated_out):
        a = generated_out['out_features']
        p = generated_out['id_features']
        neg_idx = torch.cat([torch.arange(1, a.size(-1)), torch.tensor([0])])
        n = p[..., neg_idx]

        loss = {}
        loss['point_loss'] = self.triplet_loss(a, p, n, meshwise=False, margin=self.opt.margin_point) * self.opt.lambda_point
        return loss

    def triplet_loss(self, a, p, n, meshwise, margin):
        dist_diff = torch.norm(a - p, dim=1) - torch.norm(a - n, dim=1)
        if meshwise:
            dist_diff = torch.mean(dist_diff, dim=-1)
        return torch.mean(F.relu(dist_diff + margin))

    def nearest_neigbour_distance(self, s1, s2, reduction):
        s1_ = s1.unsqueeze(0)
        s2_ = s2.unsqueeze(0)
        pairwise_dist = torch.cdist(s1_, s2_) ** 2
        out = torch.topk(pairwise_dist, k=1, dim=-1, largest=False)[0]
        if reduction == 'mean':
            return out.mean()
        if reduction == 'sum':
            return out.sum()

    def chamfer_distance(self, s1, s2, reduction):
        """
        s1: n1 x d array
        s2: n2 x d array
        """
        assert s1.shape[1] == s2.shape[1]
        distance1 = self.nearest_neigbour_distance(s1, s2, reduction)
        distance2 = self.nearest_neigbour_distance(s2, s1, reduction)
        return distance1 + distance2
