import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.generator import AdaptiveFeatureGenerator
import util.util as util

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad1d(padding)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.in1 = nn.InstanceNorm1d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad1d(padding)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.in2 = nn.InstanceNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        out = self.prelu(out)
        return out

class Correspondence(BaseNetwork):

    def __init__(self, opt):
        self.opt = opt
        super().__init__()

        self.adaptive_feature = AdaptiveFeatureGenerator(opt)

        self.feature_channel = 64
        self.layer = [
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=1, padding=0, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=1, padding=0, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=1, padding=0, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=1, padding=0, stride=1)]

        self.out_features = self.feature_channel * 4

        self.layer = nn.Sequential(*self.layer)

        self.ot_dims = torch.arange(self.out_features)
        self.in_channels = self.out_features
        self.inter_channels = 256

        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, pose_points, identity_points, return_Tm=False, encode=False, decode=False, id_features=None, pose_features=None):
        assert not (encode and decode)
        corr_out = {}

        # Extract feature
        if decode:
            assert id_features is not None
            assert pose_features is not None
            assert pose_points is not None
        else:
            pose_features= self.adaptive_feature(pose_points)
            pose_features = util.feature_normalize(pose_features)
            pose_features = self.layer(pose_features)
            if encode:
                return pose_features
            id_features= self.adaptive_feature(identity_points)
            id_features = util.feature_normalize(id_features)
            id_features = self.layer(id_features)
        corr_out['pose_features'] = pose_features
        corr_out['id_features'] = id_features

        # Correlation matrix C (cosine similarity)
        theta = self.theta(id_features[:, self.ot_dims])
        theta = torch.div(theta, torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon)

        phi = self.phi(pose_features[:, self.ot_dims])
        phi = torch.div(phi, torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon)

        C_Matrix = torch.matmul(theta.permute(0, 2, 1), phi)

        # Optimal Transport
        K = torch.exp(-(1.0 - C_Matrix) / 0.03)

        # Init. of Sinkhorn algorithm
        power = 1#gamma / (gamma + epsilon)
        a = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=theta.device, dtype=theta.dtype
            )
            / K.shape[1]
        )
        prob1 = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=theta.device, dtype=theta.dtype
            )
            / K.shape[1]
        )
        prob2 = (
            torch.ones(
                (K.shape[0], K.shape[2], 1), device=phi.device, dtype=phi.dtype
            )
            / K.shape[2]
        )

        # Sinkhorn algorithm
        for _ in range(5):
            # Update b
            KTa = torch.bmm(K.transpose(1, 2), a)
            b = torch.pow(prob2 / (KTa + 1e-8), power)
            # Update a
            Kb = torch.bmm(K, b)
            a = torch.pow(prob1 / (Kb + 1e-8), power)

        # Optimal matching matrix Tm
        T_m = torch.mul(torch.mul(a, K), b.transpose(1, 2))
        T_m = T_m / torch.sum(T_m, dim=2, keepdim=True)

        pose_for_warp = pose_points.permute(0, 2, 1)

        # Warped points
        corr_out['warp_out'] = torch.matmul(T_m, pose_for_warp)
        if return_Tm:
            corr_out['T_m_bin'] = F.one_hot(T_m.argmax(dim=2), num_classes=T_m.size(2)).to(dtype=pose_features.dtype, device=pose_features.device)

        return corr_out
