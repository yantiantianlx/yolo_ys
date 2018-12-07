
import os
import sys
import torch
import numpy as np
from collections import OrderedDict
import torch.multiprocessing as mp
mp.set_start_method('spawn')

sys.path.append(os.path.dirname(os.getcwd()))
from gear_optimize.pbt_worker_class import Worker
from gear_tool.vis import VIS


class PBT_Controller:
    def __init__(self, num_worker, pbt_save_dir='pbt_res'):
        self.num_worker = num_worker

        self.exploit_mode = 'truncation'

        self.gpu_num = torch.cuda.device_count()

        assert num_worker%self.gpu_num == 0
        self.one_step_train_num = num_worker//self.gpu_num

        self.score_dict = OrderedDict()
        self.worker_list = list()
        for i in range(num_worker):
            worker = Worker(i, save_dir=pbt_save_dir, optimize_step_num=50)
            self.worker_list.append(worker)
            self.score_dict[str(i)] = 0.0

        self.vis = VIS(show_dir=os.path.join(pbt_save_dir, 'tensorboard'))

    def exploit(self):
        if self.exploit_mode == 'truncation':
            self.score_dict = OrderedDict(sorted(self.score_dict.items(), key=lambda x: x[1]))
            exploit_num = self.num_worker//5

            delete_counter = 0
            for k, v in self.score_dict.items():
                if delete_counter >= exploit_num:
                    break
                low_score_worker_id = int(k)
                high_score_worker_id = np.random.randint(self.num_worker-exploit_num, self.num_worker)
                self.worker_list[low_score_worker_id].get_parameter_from(self.worker_list[high_score_worker_id])

                delete_counter += 1

        if self.exploit_mode == 't_test':
            pass  # TODO

    def run_one_step_of_one_worker(self, worker_id, gpu_id):
        worker = self.worker_list[worker_id]
        worker.show_hyper_by_vis()
        worker.to('cuda:'+str(gpu_id))
        worker.explore()
        worker.optimize()
        score = worker.evaluate()
        print(score)
        self.score_dict[str(worker_id)] = score
        worker.to('cpu')

    def run_one_step_of_all_worker(self):
        worker_id = 0

        for train_idx in range(self.one_step_train_num):  # gpu limited, so need many times to train

            # for gpu_id in range(self.gpu_num):
            #     self.run_one_step_of_one_worker(worker_id, gpu_id)
            #     worker_id += 1

            process_list = list()
            for gpu_id in range(self.gpu_num):
                process_list.append(mp.Process(target=self.run_one_step_of_one_worker, args=(worker_id, gpu_id,)))
                worker_id += 1

            for p in process_list:
                p.start()
                p.join()

            for p in process_list:
                p.join()

    def save_all_worker(self):
        for worker in self.worker_list:
            worker.save_state()

    def train_num(self, num):
        for i in range(num):
            self.run_one_step_of_all_worker()
            self.save_all_worker()
            self.exploit()


if __name__ == '__main__':
    pbt_controller = PBT_Controller(num_worker=4)
    pbt_controller.train_num(1000)
