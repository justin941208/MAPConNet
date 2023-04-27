from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--test_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--metric', type=str, default='PMD', help='evaluation metric', choices=['PMD', 'CD'])
        parser.add_argument('--save_output', action='store_true', help='save test outputs')
        parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')

        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
