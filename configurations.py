from model import ModelType
import numpy as np


class DefaultConfig(object):
    init_scale = 0.1
    learning_rate = 0.8
    max_grad_norm = 3
    num_layers = 1
    hidden_size = 2
    max_epoch = 5  # nr epochs with max learning rate
    max_max_epoch = 30
    keep_prob = 0.8
    lr_decay = 0.9
    batch_size = 64
    vocab_size = 29
    learning_mode = ModelType.tf


class GridSearchConfig(object):
    init_scale = 0.1
    learning_rate = np.arange(0.7, 1, 0.1)
    max_grad_norm = np.arange(3, 5, 1)
    num_layers = np.arange(1, 3, 1)
    hidden_size = 2
    max_epoch = 3  # nr epochs with max learning rate
    max_max_epoch = 15
    keep_prob = 0.8
    lr_decay = 0.9
    batch_size = [64, 128]
    vocab_size = 29
    learning_mode = ModelType.tf


class Configuration(object):
    def __init__(self):
        self.parameters = {
            "init_scale": 0.1,
            "learning_rate": np.arange(0.7, 1, 0.1),
            "max_grad_norm": np.arange(3, 5, 1),
            "num_layers": np.arange(1, 3, 1),
            "hidden_size": 2,
            "max_epoch": 3,  # nr epochs with max learning rate
            "max_max_epoch": 15,
            "keep_prob": 0.8,
            "lr_decay": 0.9,
            "batch_size": [64, 128],
            "vocab_size": 29,
            "learning_mode": ModelType.tf
        }

    @property
    def init_scale(self):
        return self.parameters["init_scale"]

    @init_scale.setter
    def init_scale(self, init_scale):
        self.parameters["init_scale"] = init_scale

    @property
    def learning_rate(self):
        return self.parameters["learning_rate"]

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.parameters["learning_rate"] = learning_rate

    @property
    def max_grad_norm(self):
        return self.parameters["max_grad_norm"]

    @max_grad_norm.setter
    def max_grad_norm(self, max_grad_norm):
        self.parameters["max_grad_norm"] = max_grad_norm

    @property
    def num_layers(self):
        return self.parameters["num_layers"]

    @num_layers.setter
    def num_layers(self, num_layers):
        self.parameters["num_layers"] = num_layers

    @property
    def hidden_size(self):
        return self.parameters["hidden_size"]

    @hidden_size.setter
    def hidden_size(self, hidden_size):
        self.parameters["hidden_size"] = hidden_size

    @property
    def max_epoch(self):
        return self.parameters["max_epoch"]

    @max_epoch.setter
    def max_epoch(self, max_epoch):
        self.parameters["max_epoch"] = max_epoch

    @property
    def max_max_epoch(self):
        return self.parameters["max_max_epoch"]

    @max_max_epoch.setter
    def max_max_epoch(self, max_max_epoch):
        self.parameters["max_max_epoch"] = max_max_epoch

    @property
    def keep_prob(self):
        return self.parameters["keep_prob"]

    @keep_prob.setter
    def keep_prob(self, keep_prob):
        self.parameters["keep_prob"] = keep_prob

    @property
    def lr_decay(self):
        return self.parameters["lr_decay"]

    @lr_decay.setter
    def lr_decay(self, lr_decay):
        self.parameters["lr_decay"] = lr_decay

    @property
    def batch_size(self):
        return self.parameters["batch_size"]

    @batch_size.setter
    def batch_size(self, batch_size):
        self.parameters["batch_size"] = batch_size

    @property
    def vocab_size(self):
        return self.parameters["vocab_size"]

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self.parameters["vocab_size"] = vocab_size

    @property
    def learning_mode(self):
        return self.parameters["learning_mode"]

    @learning_mode.setter
    def learning_mode(self, learning_mode):
        self.parameters["learning_mode"] = learning_mode
