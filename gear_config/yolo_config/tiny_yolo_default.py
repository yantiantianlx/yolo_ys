import os
import sys
import time
from os.path import join
from gear_config.yaml_to_object import Cls


config_running_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))


class ARG_DATA_DATALOADER_TRAIN(Cls):
    def __init__(self):
        super().__init__()
        self.batch_size = 128
        self.drop_last = True
        self.gear_cls_tree_path = 'arg.data.dataloader.train'
        self.num_workers = 32
        self.shuffle = True


class ARG_DATA_DATALOADER_VALID(Cls):
    def __init__(self):
        super().__init__()
        self.batch_size = 1
        self.gear_cls_tree_path = 'arg.data.dataloader.valid'
        self.num_workers = 4
        self.shuffle = True


class ARG_DATA_DATALOADER(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.data.dataloader'
        self.train = ARG_DATA_DATALOADER_TRAIN()
        self.valid = ARG_DATA_DATALOADER_VALID()


class ARG_DATA_DATASET_TEST(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.data.dataset.test'
        self.root = '/simple_ssd/ys2/ys_MSCOCO/test'


class ARG_DATA_DATASET_TRAIN(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.data.dataset.train'
        self.max_detection_num = 50
        self.root = '/simple_ssd/ys2/ys_MSCOCO/train'
        self.txt_path = None


class ARG_DATA_DATASET(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.data.dataset'
        self.test = ARG_DATA_DATASET_TEST()
        self.train = ARG_DATA_DATASET_TRAIN()


class ARG_DATA(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.data'
        self.dataloader = ARG_DATA_DATALOADER()
        self.dataset = ARG_DATA_DATASET()


class ARG_MODEL_NET(Cls):
    def __init__(self):
        super().__init__()
        self.feature_size = [[13, 13], [26, 26]]
        self.gear_cls_tree_path = 'arg.model.net'
        self.im_size = [416, 416]
        self.num_anchor = len([[0.024038, 0.033654], [0.055288, 0.064904], [0.088942, 0.139423], [0.194712, 0.197115], [0.324519, 0.40625], [0.826923, 0.766827]])
        self.num_class = 80


class ARG_MODEL(Cls):
    def __init__(self):
        super().__init__()
        self.flt_anchor = [[0.024038, 0.033654], [0.055288, 0.064904], [0.088942, 0.139423], [0.194712, 0.197115], [0.324519, 0.40625], [0.826923, 0.766827]]
        self.gear_cls_tree_path = 'arg.model'
        self.loss = None
        self.mask_iou_threshold = 0.5
        self.nms_iou_threshold = 0.4
        self.out_confidence_filter_threshold = 0.5
        self.weight_path = None
        self.net = ARG_MODEL_NET()


class ARG_OPTIM(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.optim'
        self.lr = 0.001
        self.weight_decay = 1e-05


class ARG_SAVE(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.save'
        self.model = join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'train/res', 'tiny_yolo_default'+'_'+'ys2'+'_'+config_running_time)
        self.root = None
        self.tensorboard = join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'train/res/tensorboard', 'tiny_yolo_default'+'_'+'ys2'+'_'+config_running_time)


class ARG_TENSORBOARD(Cls):
    def __init__(self):
        super().__init__()
        self.enable = True
        self.gear_cls_tree_path = 'arg.tensorboard'


class ARG_TEST(Cls):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.device_ids = [0]
        self.gear_cls_tree_path = 'arg.test'
        self.image_dir = join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'tst/images')
        self.out_dir = join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'tst/images_out')


class ARG_TRAIN(Cls):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.device_ids = [0, 1, 2, 3]
        self.gear_cls_tree_path = 'arg.train'
        self.save_interval = 1000
        self.test_interval = 1000
        self.valid_interval = 100
        self.vis_interval = 10


class ARG(Cls):
    def __init__(self):
        super().__init__()
        self.gear_abs_config_file_name = '/home/ys/Desktop/tinyyolo/gear_config/yolo_config/tiny_yolo_default.yaml'
        self.gear_cls_tree_path = 'arg'
        self.data = ARG_DATA()
        self.model = ARG_MODEL()
        self.optim = ARG_OPTIM()
        self.save = ARG_SAVE()
        self.tensorboard = ARG_TENSORBOARD()
        self.test = ARG_TEST()
        self.train = ARG_TRAIN()


