from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=2000, help='frequency of saving training results')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of printing training losses')
        parser.add_argument('--save_latest_freq', type=int, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_training_output', action='store_true', help='save training output')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_override', type=float, help='override starting lr when resuming training')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_edge', type=float, default=0.5, help='weight for edge loss')
        parser.add_argument('--lambda_rec', type=float, default=1000.0, help='weight for rec loss')
        parser.add_argument('--lambda_mesh', type=float, default=1.0, help='weight for mesh loss')
        parser.add_argument('--lambda_point', type=float, default=1.0, help='weight for point loss')
        parser.add_argument('--margin_mesh', type=float, default=1.0, help='margin for mesh loss')
        parser.add_argument('--margin_point', type=float, default=1.0, help='margin for point loss')

        # additional options
        parser.add_argument('--percentage', type=int, default=100, choices=[0, 50, 100], help='percentage of identities and poses in training data relative to default')
        parser.add_argument('--use_unlabelled', action='store_true', help='load meshes without ground truth target mesh')

        self.isTrain = True
        return parser
