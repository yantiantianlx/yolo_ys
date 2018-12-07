import os


def prepare_save_dirs(arg):
    attr_list = dir(arg.save)
    attr_list = list(filter(lambda x: not (x[:1] == '_' or x == 'cover_by' or x == ''), attr_list))
    for path_name in attr_list:
        path = getattr(arg.save, path_name)
        if isinstance(path, str) and os.path.isabs(path) and not os.path.exists(path):
            os.makedirs(path, mode=0o777)

