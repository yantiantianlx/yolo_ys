import os
import re

if __package__ is None or __package__ == '':
    from yaml_to_object import get_Cls, Cls
    from gear_cls_str_decoder import Decoder
else:
    from .yaml_to_object import get_Cls, Cls
    from .gear_cls_str_decoder import Decoder


def is_leaf_cls(obj, attr_list):
    if len(attr_list) == 0:
        return False
    for attr in attr_list:
        if isinstance(getattr(obj, attr), Cls):
            print('False')
            return False
    print('True')
    return True


class Writer:
    def __init__(self):
        self.function_str = '(?:[_a-zA-Z]\w*(?:\\.[_a-zA-Z]\w*)*\\(.*\\))|\\+'
        self.base_function_pattern = re.compile(self.function_str)

        self.lines = []
        self.lines = []
        line = "import os"
        self.lines.append(line)
        line = "import sys"
        self.lines.append(line)
        line = "import time"
        self.lines.append(line)
        line = "from os.path import join"
        self.lines.append(line)
        line = "from gear_config.yaml_to_object import Cls"
        self.lines.append(line)
        line = ""
        self.lines.append(line)
        line = ""
        self.lines.append(line)
        line = "config_running_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))"
        self.lines.append(line)
        line = ""
        self.lines.append(line)
        line = ""
        self.lines.append(line)

    def write_down_obj(self, obj):
        if not isinstance(obj, Cls):
            return
        attr_list = dir(obj)
        attr_list = list(filter(lambda x: not (x[:1] == '_' or callable(getattr(obj, x)) or x == ''), attr_list))
        for attr in attr_list:
            self.write_down_obj(getattr(obj, attr))

        Cls_name = obj.gear_cls_tree_path.replace('.', '_').upper()
        line = "class {}(Cls):".format(Cls_name)
        self.lines.append(line)
        line = "    def __init__(self):"
        self.lines.append(line)
        line = "        super().__init__()"
        self.lines.append(line)

        cls_list = list()
        build_in_list = list()
        for attr in attr_list:
            if isinstance(getattr(obj, attr), Cls):
                cls_list.append(attr)
            else:
                build_in_list.append(attr)

        for attr in build_in_list:
            context = getattr(obj, attr)
            if isinstance(context, str) and self.base_function_pattern.search(context) is None:
                line = "        self.{} = '{}'".format(attr, getattr(obj, attr))
                self.lines.append(line)
            else:
                line = "        self.{} = {}".format(attr, getattr(obj, attr))
                self.lines.append(line)

        for attr in cls_list:
            attr_Cls_name = getattr(obj, attr).gear_cls_tree_path.replace('.', '_').upper()
            line = "        self.{} = {}()".format(attr, attr_Cls_name)
            self.lines.append(line)

        self.lines.append("\n")


cwd = os.getcwd()

yaml_wait_list = []

for fpathe, dirs, fs in os.walk(cwd):
    for f in fs:
        if f[-5:] == '.yaml':
            yaml_file = os.path.join(fpathe, f)
            yaml_wait_list.append(yaml_file)


for yaml_file in yaml_wait_list:
    py_file_path = yaml_file[:-5]+'.py'

    arg = get_Cls(yaml_file)
    arg = Decoder(arg).decode()

    writer = Writer()
    writer.write_down_obj(arg)

    with open(py_file_path, "w") as f:
        for line in writer.lines:
            f.write(line)
            f.write('\n')


