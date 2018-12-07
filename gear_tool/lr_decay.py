import numpy as np
import math


class CosineDecayRestarts:
    def __init__(self, learning_rate, first_decay_steps, t_mul=2.0, m_mul=1.0, alpha=0.0):
        self.decay_steps = first_decay_steps
        self.round_counter = 0
        self.last_explore_step = 0
        self.max_lr = learning_rate

        self.lr = learning_rate

        def lr_func(x):
            reminder_steps = x - self.last_explore_step
            if reminder_steps > self.decay_steps:
                self.decay_steps = int(self.decay_steps * t_mul)
                self.max_lr = self.max_lr * m_mul
                self.round_counter += 1
                self.last_explore_step = x
                reminder_steps = 0
            cosine_decay = 0.5 * (1 + math.cos(math.pi * reminder_steps / self.decay_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            decayed_learning_rate = self.max_lr * decayed
            return decayed_learning_rate

        self.lr_func = lr_func

    def update(self, global_step):
        self.lr = self.lr_func(global_step)

    def get_lr(self, global_step):
        self.update(global_step)
        return self.lr


class ExponentialDecay:
    def __init__(self, learning_rate, decay_steps, decay_rate, staircase=False):
        assert decay_rate < 1.0

        self.lr = learning_rate

        if staircase:
            pass  # TODO
            # def lr_func(x):
            #     x_floor = (x//decay_steps)*decay_steps
            #     decayed_learning_rate = learning_rate * decay_rate ** (x_floor / decay_steps)
            #     return decayed_learning_rate
        else:
            def lr_func(x):
                if x > decay_steps:
                    x = decay_steps
                decayed_learning_rate = learning_rate * decay_rate ** (x / decay_steps)
                return decayed_learning_rate

        self.lr_func = lr_func

    def update(self, global_step):
        self.lr = self.lr_func(global_step)

    def get_lr(self, global_step):
        self.update(global_step)
        return self.lr


class Constant:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, global_step):
        pass

    def get_lr(self, global_step):
        return self.lr


class LrSeeker:
    def __init__(self, min_lr=1e-10, max_lr=1e-0, sample_step=None):
        lg_min_lr = int(math.log(min_lr, 10))
        lg_max_lr = int(math.log(max_lr, 10))

        if sample_step is None:
            step_len = 0.1
        else:
            step_len = (lg_max_lr-lg_min_lr)//sample_step

        self.samples = np.arange(lg_min_lr, lg_max_lr, step_len)
        self.samples_lr = 10**self.samples

    def get_lr(self, global_step):
        if global_step > len(self.samples_lr):
            global_step = len(self.samples_lr)-1
            print('learning rate search end!')
        lr = self.samples_lr[global_step]
        return lr


if __name__ == '__main__':
    from tqdm import tqdm

    lr_decay = ExponentialDecay(0.1, 10000, 1e-7)

    # lr_decay = LrSeeker()

    # optimizer.param_groups[0]['lr'] =
    import matplotlib.pyplot as plt

    x = np.arange(0, 10000)
    y = np.ndarray((10000,))
    for i in tqdm(x):
        a = lr_decay.get_lr(i)
        y[i] = a

    plt.plot(x, np.log10(y))
    plt.show()

