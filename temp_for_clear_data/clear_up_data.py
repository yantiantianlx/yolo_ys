import os
import shutil
import time
import re


def corresponding_json_path(image_path):
    file_, extension = split_extension_name(image_path)
    json_path = file_ + '.json'
    if os.path.exists(json_path):
        return json_path
    father_dir = os.path.dirname(image_path)
    grand_pa_dir = os.path.dirname(father_dir)
    father_dir_list = []
    for rela_path in os.listdir(grand_pa_dir):
        abs_path = os.path.join(grand_pa_dir, rela_path)
        if os.path.isdir(abs_path):
            father_dir_list.append(abs_path)
    assert len(father_dir_list) == 2, 'rearrange you file dir, image and json is not pair!'
    father_dir_list.remove(father_dir)
    uncle_dir = father_dir_list[0]
    base_file_, _ = split_extension_name(os.path.basename(image_path))
    json_path = os.path.join(uncle_dir, base_file_ + '.json')
    return json_path


def split_extension_name(name):
    extension_pattern = re.compile('\\.[a-zA-z]*$')
    search = extension_pattern.search(name)
    assert search is not None, 'wrong file! at name: {}'.format(name)
    search_start = search.regs[0][0]
    return name[:search_start], name[search_start:]


class Clear_Tool:
    def __init__(self, abs_path: str, output_dir=None, output_mode='dir'):
        self.abs_path = abs_path
        self.output_mode = output_mode
        if output_dir is None:
            base_name = os.path.basename(abs_path)
            self.output_dir = os.path.join(os.path.dirname(abs_path), 'cleared_'+base_name)
        self.generate_time_str = time.strftime('%Y_%m_%d', time.localtime(time.time()))
        self.dfs(abs_path)

    def dfs(self, father_abs_path):
        for rela_path in os.listdir(father_abs_path):
            abs_path = os.path.join(father_abs_path, rela_path)
            if os.path.isfile(abs_path):
                file_, extension = split_extension_name(abs_path)
                assert extension in ['.json', '.jpg', '.png', '.jpeg'], 'wrong file! at path: {}'.format(abs_path)
                if extension in ['.jpg', '.png', '.jpeg']:
                    image_path = abs_path
                    json_path = corresponding_json_path(image_path)
                    self.save_pair_to_output_dir(image_path, json_path)

            if os.path.isdir(abs_path):
                self.dfs(abs_path)

    def save_pair_to_output_dir(self, abs_image_file, abs_json_path):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(os.path.join(self.output_dir, 'json')):
            os.mkdir(os.path.join(self.output_dir, 'json'))
        if not os.path.exists(os.path.join(self.output_dir, 'image')):
            os.mkdir(os.path.join(self.output_dir, 'image'))

        image_name = os.path.basename(abs_image_file)
        image_, image_extension = split_extension_name(image_name)
        json_name = os.path.basename(abs_json_path)
        json_, json_extension = split_extension_name(json_name)

        i = 0
        while True:
            new_image_name = self.generate_time_str+'_'+image_+'_'+'{:0>3}'.format(i)+image_extension
            output_image_path = os.path.join(self.output_dir, 'image', new_image_name)
            new_json_name = self.generate_time_str+'_'+json_+'_'+'{:0>3}'.format(i)+json_extension
            output_json_path = os.path.join(self.output_dir, 'json', new_json_name)

            print(output_image_path)

            if not os.path.exists(output_image_path):
                shutil.copyfile(abs_image_file, output_image_path)
                assert not os.path.exists(output_json_path)
                shutil.copyfile(abs_json_path, output_json_path)
                break
            i += 1



if __name__ == '__main__':
    test_dir = '/simple_ssd/ys2/tiny_yolo_project/hand_detection'
    clear = Clear_Tool(test_dir)