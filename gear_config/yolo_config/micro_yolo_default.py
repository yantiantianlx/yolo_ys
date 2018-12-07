import os
import sys
import time
from os.path import join
from gear_config.yaml_to_object import Cls


config_running_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))


class ARG_DATA_DATALOADER_TRAIN(Cls):
    def __init__(self):
        super().__init__()
        self.batch_size = 16
        self.gear_cls_tree_path = 'arg.data.dataloader.train'
        self.num_workers = 16
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


class ARG_DATA_DATASET_TRAIN(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.data.dataset.train'
        self.relative_txt_path = None
        self.root = '/simple_ssd/ys2/tiny_yolo_project/cleared_hand_detection'
        self.txt_path = None


class ARG_DATA_DATASET(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.data.dataset'
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
        self.feature_size = [[10, 10], [20, 20]]
        self.gear_cls_tree_path = 'arg.model.net'
        self.im_size = [160, 160]
        self.num_anchor = len([[10, 14], [23, 27], [37, 58], [81, 82], [135, 169]])
        self.num_class = 1


class ARG_MODEL(Cls):
    def __init__(self):
        super().__init__()
        self.anchor = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169]]
        self.gear_cls_tree_path = 'arg.model'
        self.loss = None
        self.mask_iou_threshold = 0.5
        self.nms_iou_threshold = 0.4
        self.out_confidence_filter_threshold = 0.05
        self.weight_path = join(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))), 'model/body_micro_yolo_epoch1_weight')
        self.net = ARG_MODEL_NET()


class ARG_OPTIM(Cls):
    def __init__(self):
        super().__init__()
        self.gear_cls_tree_path = 'arg.optim'
        self.lr = 0.0001
        self.weight_decay = 1e-05


class ARG_SAVE(Cls):
    def __init__(self):
        super().__init__()
        self.analyze = join(join('/simple_ssd/ys2/tiny_yolo_project/tiny_yolo/train/res', 'micro_yolo_default'+'_'+'ys2'+'_'+config_running_time), 'valid/coco_analyze')
        self.gear_cls_tree_path = 'arg.save'
        self.model = join(join('/simple_ssd/ys2/tiny_yolo_project/tiny_yolo/train/res', 'micro_yolo_default'+'_'+'ys2'+'_'+config_running_time), 'models')
        self.root = join('/simple_ssd/ys2/tiny_yolo_project/tiny_yolo/train/res', 'micro_yolo_default'+'_'+'ys2'+'_'+config_running_time)
        self.tensorboard = join(join('/simple_ssd/ys2/tiny_yolo_project/tiny_yolo/train/res', 'micro_yolo_default'+'_'+'ys2'+'_'+config_running_time), 'tensorboard')
        self.test = join(join('/simple_ssd/ys2/tiny_yolo_project/tiny_yolo/train/res', 'micro_yolo_default'+'_'+'ys2'+'_'+config_running_time), 'tst')
        self.train = join(join('/simple_ssd/ys2/tiny_yolo_project/tiny_yolo/train/res', 'micro_yolo_default'+'_'+'ys2'+'_'+config_running_time), 'train')
        self.valid = join(join('/simple_ssd/ys2/tiny_yolo_project/tiny_yolo/train/res', 'micro_yolo_default'+'_'+'ys2'+'_'+config_running_time), 'valid')


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
        self.epochs = 100
        self.gear_cls_tree_path = 'arg.train'
        self.log_iteration_interval = 10
        self.save_model_epoch_interval = 1
        self.valid_epoch_interval = 1


class ARG(Cls):
    def __init__(self):
        super().__init__()
        self.gear_abs_config_file_name = '/home/ys/Desktop/tinyyolo/gear_config/yolo_config/micro_yolo_default.yaml'
        self.gear_cls_tree_path = 'arg'
        self.data = ARG_DATA()
        self.model = ARG_MODEL()
        self.optim = ARG_OPTIM()
        self.save = ARG_SAVE()
        self.tensorboard = ARG_TENSORBOARD()
        self.test = ARG_TEST()
        self.train = ARG_TRAIN()


