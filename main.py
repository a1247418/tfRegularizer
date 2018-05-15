"""
Based on the official tutorial:
https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/*.py
"""

import os
import time
import copy
from enum import Enum
import numpy as np
import tensorflow as tf
import dataReader
from model import SeqModel, ModelType, ModelInput, Mode
from configurations import *
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python import debug as tf_debug


PRINT_OUT = True
DATA_FROM_SAVES = True

np.random.seed(1)
tf.set_random_seed(1)

path = os.getcwd()
data_path = path + os.sep + "data"
save_path = path + os.sep + "saves"
logs_path = path + os.sep + "logs"

flags = tf.flags
logging = tf.logging

# TODO: actually use datatype
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")


def data_type():
    return tf.float16 if tf.flags.FLAGS.use_fp16 else tf.float32


def get_time_str():
    return time.strftime("%d.%m. %H:%M:%S", time.gmtime())


def run_epoch(sess, model, writer=None, summary_op=None, eval_op=None):
    """Run throught the whole dataset once."""
    if PRINT_OUT: print("Start running epoch.")

    sess.run(model.dataset.iterator.initializer)

    # start_time = get_time_str()
    costs = 0.0
    total_steps = 0

    # Operations/tensors to evaluate
    fetches = {
        "cost": model.cost,
        "num_steps_in_batch": model._num_steps_in_batch,
        "final_state": model.final_state,

        "loss": model._loss,
        "mask": model._mask,
        "targets": model._target,
        "logits": model._logits
    }
    # Evaluate other operations, if requested
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    if summary_op is not None:
        fetches["summary_op"] = summary_op
    step = 0
    while True:
        try:
            # Fetch next batch
            example_batch = sess.run(model.dataset.next_element)
        except tf.errors.OutOfRangeError:
            print("Processed", step, "batches.")
            break

        # num_steps = example_batch.shape[1]
        batch_size = example_batch.shape[0]

        state = sess.run(model.initial_state, {model.batch_size: batch_size})

        feed_dict = {model.batch: example_batch,
                     model.batch_size: batch_size}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = sess.run(fetches, feed_dict)
        cost = vals["cost"]
        num_steps_in_batch = vals["num_steps_in_batch"]
        # state = vals["final_state"]

        costs += cost
        total_steps += num_steps_in_batch

        if step > 0 and step % 500 == 0:
            '''
            print(vals["targets"])
            print(np.argmax(vals["logits"], axis=2))
            print(vals["mask"])
            print(vals["loss"])
            '''

            print(get_time_str(), ": Batch %.3f, perplexity: %.3f" %
                  (step, np.exp(costs/total_steps)))
            if summary_op is not None:
                # writer = SummaryWriterCache.get(logs_path)
                writer.add_summary(vals["summary_op"])
                writer.flush()

        step += 1

        # summary_writer = SummaryWriterCache.get(logs_path)
        # summary_writer.add_summary(vals["summary"])

    return np.exp(costs/total_steps)


def instantiate_model(mode, config, data_set, reuse):
    """
    Instantiates the model with the given configuration and data.
    :param mode: Mode enum specifying train/test/validation.
    :param config:
    :param data_set: a tf.data.dataset
    :param reuse: reuse graph
    :return: model, summary operation
    """
    name = Mode.to_string(mode)
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope(name):
        model_input = ModelInput(
            config=config,
            data=data_set,
            name=name+"Input")
        with tf.variable_scope("Model", reuse=reuse, initializer=initializer):
            model = SeqModel(
                is_training=(mode == Mode.train),
                config=config,
                dataset=model_input)
            summary = tf.summary.scalar(name+" Loss", model.normalized_cost)

            if mode == Mode.train:
                summ_lr = tf.summary.scalar("Learning Rate", model.lr)
                summary = tf.summary.merge([summary, summ_lr])

    return model, summary


def search_params(config, train_set, validation_set):
    """
    Performs grid search over the value ranges in the reference configuration.
    :param config: Reference configuration with ranges as parameter values.
    :param train_set: Dataset to use for training.
    :param validation_set: Dataset to use for validation.
    :return: Configuration with smallest validation loss.
    """
    #Generate configs to be tested

    configs = []
    score = []

    best_config = None
    best_loss = float("inf")

    for config in configs:
        mtrain, train_summary_op = instantiate_model(
            mode=Mode.train,
            config=config,
            data_set=train_set,
            reuse=True)
        run_loop([Mode.train], [mtrain], [train_summary_op], log=True, save=False)

        validation_config = copy.deepcopy(config)
        validation_config.max_max_epoch = 1

        mvalid, val_summary_op = instantiate_model(
            mode=Mode.validate,
            config=validation_config,
            data_set=validation_set,
            reuse=False)
        config_loss = run_loop([Mode.train], [mtrain], [train_summary_op], log=True, save=False)[0][0]

        if best_loss > config_loss:
            best_config = config
            best_loss = config_loss

    return  best_config

def run_loop(modes, models, summary_ops=None, log=False, save=False):
    """
    Runs models for the specified number of epochs.
    :param modes: List of modes. Whether to train, evaluate, or test
    :param models: List of models
    :param summary_ops: List of summary operations with one operation per model, or None.
    :param log: Whether to write summaries for tensorboard.
    :param save: Whether to write checkpoints and save the final models.
    :return: List of lists, where each sub-list is the loss/epoch.
    """
    assert (len(modes) == len(models))
    assert (summary_ops is None or len(summary_ops) == len(models))
    assert (summary_ops is None or log == True)

    nr_models = len(modes)
    representative_config = models[0].config  # used for variables that are assumed to be the same between models

    # Saver hook that allows the monitored session to automatically create checkpoints
    saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=save_path, save_steps=10000) if save else None
    # summary_hook = tf.train.SummarySaverHook(output_dir=logs_path, summary_op=train_summary_op, save_secs=10)
    with tf.train.MonitoredTrainingSession(hooks=[saver_hook]) as session:
        # session = tf_debug.LocalCLIDebugWrapperSession(session)  # Enable debug

        writers = []
        losses = []
        for i in range(nr_models):
            losses.append([])
            if log:
                writers.append(
                    tf.summary.FileWriter(logs_path + os.sep + Mode.to_string(modes[i]).lower(), session.graph))
            else:
                writers.append(None)

        for i in range(representative_config.max_max_epoch):
            lr_decay = representative_config.lr_decay ** max(i + 1 - representative_config.max_epoch, 0)

            for m in range(nr_models):
                model = models[m]
                model.assign_lr(session, representative_config.learning_rate * lr_decay)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))
                perplexity = run_epoch(session, model,
                                       writer=writers[m],
                                       summary_op=summary_ops[m],
                                       eval_op=model.train_op if modes[m] == Mode.train else None)
                losses[m].append(perplexity)
                print("Epoch: %d ", modes[m], " Perplexity: %.3f" % (i + 1, perplexity))

        if save:
            print("Saving model to %s." % save_path)
            saver = tf.train.Saver()
            saver.save(session, save_path,
                       global_step=tf.train.global_step(session, tf.train.get_or_create_global_step()))

        return losses


def main(_):
    train_set, validation_set, test_set, vocab, char2int = \
        dataReader.loadDataFromSaves() if DATA_FROM_SAVES else \
        dataReader.loadDataFromSource()
    config = DefaultConfig()
    config.vocab_size = len(vocab) + (0 not in vocab)  # Add padding token, if not already used

    with tf.Graph().as_default():
        config = search_params(GridSearchConfig(), train_set, validation_set)

        if(False):  # Train & test
            mtrain, train_summary_op = instantiate_model(
                mode=Mode.train,
                config=config,
                data_set=train_set,
                reuse=True)

            mvalid, val_summary_op = instantiate_model(
                mode=Mode.validate,
                config=config,
                data_set=validation_set,
                reuse=False)

            mtest, test_summary_op = instantiate_model(
                mode=Mode.test,
                config=config,
                data_set=test_set,
                reuse=False)

            run_loop([Mode.train], [mtrain], [train_summary_op], log=True, save=True)

            run_loop([Mode.test], [mtest], [test_summary_op])


if __name__ == "__main__":
    tf.app.run()

# tensorboard --logdir=D:/Projekte/TFRegularizer/tfRegularizer/logs
# http://0.0.0.0:6006/
