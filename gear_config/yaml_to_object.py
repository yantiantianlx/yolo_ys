import yaml


def input_parsing(inp):
    if isinstance(inp, str):
        path = inp
        f = open(path)
        d = yaml.load(f)
    else:
        raise Exception('default input type not support')
    return d


def dic2obj(d, gear_cls_tree_path, config_file_name=None):
    cls = Cls()
    cls.gear_cls_tree_path = gear_cls_tree_path
    if gear_cls_tree_path == 'arg':
        cls.gear_abs_config_file_name = config_file_name
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(cls, k, dic2obj(v, gear_cls_tree_path+'.'+k))
        else:
            setattr(cls, k, v)
    return cls


def obj_eq(ob_0, ob_1, no_list=None):
    if no_list is None:
        no_list = []
    attr_list = dir(ob_1)
    attr_list = list(filter(lambda x: not (x[:1] == '_' or x == 'cover_by' or x == ''), attr_list))
    for attr in attr_list:
        if attr in no_list:
            continue
        if hasattr(ob_0, attr) ^ hasattr(ob_1, attr):
            return False
        if not isinstance(getattr(ob_0, attr), Cls):
            if getattr(ob_0, attr) != getattr(ob_1, attr):
                return False
        if isinstance(getattr(ob_0, attr), Cls):
            return obj_eq(getattr(ob_0, attr), getattr(ob_1, attr))

    return True


def is_obj_no_none(obj):
    attr_list = dir(obj)
    attr_list = list(filter(lambda x: not (x[:1] == '_' or x == 'cover_by' or x == ''), attr_list))
    for attr in attr_list:
        if not isinstance(getattr(obj, attr), Cls):
            if getattr(obj, attr) is None:
                print(attr, 'is None!')
                return False
        else:
            if not is_obj_no_none(getattr(obj, attr)):
                print(attr, 'is None!')
                return False
    return True


def merge(a, b):
    assert isinstance(a, Cls)
    assert isinstance(b, Cls)
    """
    :param a: the default gear_config object
    :param b: the specific gear_config object
    :return: a covered by b
    """
    b_attr_list = dir(b)
    b_attr_list = list(filter(lambda x: not x[:1] == '_', b_attr_list))
    for attr in b_attr_list:
        if not isinstance(getattr(b, attr), Cls):
            val = getattr(b, attr)
            setattr(a, attr, val)
        if isinstance(getattr(b, attr), Cls):
            merge(getattr(a, attr), getattr(b, attr))
    return a


class Cls:
    def __init__(self):
        self.gear_cls_tree_path = 'arg'

    def __eq__(self, other):
        return obj_eq(self, other)

    def __ne__(self, other):
        return not obj_eq(self, other)

    def cover_by(self, other):
        merge(self,other)


def get_Cls(default_file):
    d = input_parsing(default_file)
    obj = dic2obj(d, 'arg', default_file)
    return obj

