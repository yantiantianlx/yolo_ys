import os
import numpy as np
from torch import nn
from copy import copy as deep_copy
from train.trainer_body_yolo import Trainer
from gear_tool.vis import VIS


class Worker(nn.Module):
    def __init__(self,
                 worker_id,
                 save_dir,
                 explore_mode='perturb',
                 optimize_step_num=1e4,):
        super().__init__()
        self.id = worker_id
        self.save_dir = save_dir
        self.train_generation = 0

        self.vis = VIS(save_dir=os.path.join(self.save_dir, 'tensorboard', 'worker'+str(self.id)))
        self.trainer = Trainer(vis=self.vis, id=self.id)

        self.score = 0.0
        self.explore_mode = explore_mode
        self.optimize_step_num = int(optimize_step_num)

    def optimize(self):
        self.trainer.train_steps(num_step=self.optimize_step_num)
        self.train_generation += 1

    def evaluate(self):
        self.score = self.trainer.evaluate()
        self.vis.line('2_test_AP', y=self.score)
        return self.score

    def get_parameter_from(self, high_score_worker):  # for exploit
        self.trainer.hyper_parameter = deep_copy(high_score_worker.trainer.hyper_parameter)
        self.trainer.net = deep_copy(high_score_worker.trainer.net)

    def explore(self):
        if self.explore_mode == 'perturb':
            for k, v in self.trainer.hyper_parameter.items():
                self.trainer.hyper_parameter[k] = v * (1 + np.random.uniform(-0.2, 0.2))

        if self.explore_mode == 'resample':
            pass  # TODO

    def save_state(self):
        weight_save_path = os.path.join(self.save_dir, 'step'+str(self.train_generation*self.optimize_step_num), 'worker'+str(self.id)+'weight')
        hyper_save_path = os.path.join(self.save_dir, 'step'+str(self.train_generation*self.optimize_step_num), 'worker'+str(self.id)+'hyper')

        self.trainer.save_weight(weight_save_path)
        self.trainer.save_hyper_parameter(hyper_save_path)

    def show_hyper_by_vis(self):
        for k, v in self.trainer.hyper_parameter.items():
            self.vis.line('pbt_hyper/'+k, y=v)

    # def load_state(self):
    #     torch.save(self.w, self.weight_save_path)
    #     with open(self.hyper_save_path, 'r') as f:
    #         self.h = json.load(f)
    #     return self.w, self.h
