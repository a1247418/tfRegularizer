"""
Based on the official tutorial:
https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/*.py
"""

import numpy as np
import tensorflow as tf
import dataReader
from configurations import Config, DefaultConfig, GridSearchConfig
from utils import save_path, log_path, data_path, log
import trainingControl as tCtrl
import paramSearch as pSearch

DATA_FROM_SAVES = True

np.random.seed(1)
tf.set_random_seed(1)

flags = tf.flags

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

flags.DEFINE_integer("nr_threads", 16, "Nr of treads to spawn.")


def data_type():
    return tf.float16 if tf.flags.FLAGS.use_fp16 else tf.float32


def main(_):
    train_set, validation_set, test_set, vocab, char2int = \
        dataReader.load_data_from_saves() if DATA_FROM_SAVES else \
        dataReader.load_data_from_source()

    log("Train set size: %d" % len(train_set))
    log("Validation set size: %d" % len(validation_set))
    log("Test set size: %d" % len(test_set))
    log("Vocabulary size: %d" % len(vocab))

    config = DefaultConfig()
    config.vocab_size = len(vocab) + (0 not in vocab)  # Add padding token, if not already used

    search_nr_epochs = 20
    nr_epochs = 30

    search_config = GridSearchConfig()

    search_config.nr_epochs = search_nr_epochs

    config = pSearch.search_params_parallel(search_config, train_set, validation_set)

    config.nr_epochs = nr_epochs

    tCtrl.train_and_test(config=config, train_set=train_set+validation_set, test_set=test_set, name_modifier="Final")


if __name__ == "__main__":
    tf.app.run()

# tensorboard --logdir=D:/Projekte/TFRegularizer/tfRegularizer/logs
# http://0.0.0.0:6006/
