from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=10,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain '
                                      'number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=100,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, '
                                      '<epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--epochs_warmup', type=int, default=100, help='# of epoch to  linearly warm up lr')
        self.parser.add_argument('--epochs_anneal', type=int, default=100,
                                 help='# of epoch to linearly decay learning rate to zero')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--gradient_penalty', default=True, help='use gradient_penality')
        self.parser.add_argument('--lambda_gp', type=float, default=0.1 / 100 ** 2, help='weight for gradient penalty')
        self.parser.add_argument('--constant', type=float, default=100, help='constant for gradient penalty')
        self.parser.add_argument('--gan_mode', default='lsgan', help='lsgan loss ')
        self.parser.add_argument('--lambda_AB', type=float, default=10.0, help='weight for gc loss')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')
        # self.parser.add_argument('--no_html', action='store_true',
        #                          help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='linear',
                                 help='learning rate policy: linear ')
        self.parser.add_argument('--geometry', type=str, default='rot', help='pre-defined geometry transformation '
                                                                             'function: rot|vf|hf')


        self.parser.add_argument('--lr_decay_iters', type=int, default=50,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--identity', type=float, default=0.5,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling '
                                      'the weight of the identity mapping loss. For example, if the weight of the '
                                      'identity loss should be 10 times smaller than the weight of the reconstruction '
                                      'loss, please set optidentity = 0.1')
        self.parser.add_argument('--lambda_gc', type=float, default=2.0, help='trade-off parameter for Gc and idt')
        self.parser.add_argument('--lambda_G', type=float, default=1.0, help='trade-off parameter for G, gc, and idt')

        self.isTrain = True
