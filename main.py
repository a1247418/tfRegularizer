"""
Based on the official tutorial:
https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/*.py
"""

import os
import copy
import collections
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
import dataReader
import utils
from model import SeqModel, ModelType, ModelInput, Mode
from configurations import Config, DefaultConfig, GridSearchConfig
from utils import save_path, log_path, data_path, log

DATA_FROM_SAVES = True

np.random.seed(1)
tf.set_random_seed(1)

flags = tf.flags

# TODO: actually use datatype
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")


def data_type():
    return tf.float16 if tf.flags.FLAGS.use_fp16 else tf.float32


def run_epoch(sess, model, writer=None, summary_op=None, eval_op=None):
    """Run throught the whole dataset once."""
    sess.run(model.dataset.iterator.initializer)

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
            # log("Batch %d, perplexity: %.3f" % (step, np.exp(costs/total_steps)))
            # Break out of loop if there is no more batch in the dataset
            break

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

        costs += cost
        total_steps += num_steps_in_batch

        if step % 1000 == 0:
            # log("Batch %d, perplexity: %.3f" % (step, np.exp(costs/total_steps)))
            if summary_op is not None:
                writer.add_summary(vals["summary_op"])
                writer.flush()

        step += 1

    return np.exp(costs/total_steps)


def instantiate_model(mode, config, data_set):
    """
    Instantiates the model with the given configuration and data.
    :param mode: Mode enum specifying train/test/validation.
    :param config:
    :param data_set: a tf.data.dataset
    :return: model, summary operation
    """
    name = Mode.to_string(mode)

    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope(name=name):
        model_input = ModelInput(
            config=config,
            data=data_set)
        with tf.variable_scope("Model", reuse=tf.AUTO_REUSE, initializer=initializer):
            model = SeqModel(
                is_training=(mode == Mode.train),
                config=config,
                dataset=model_input)

            summary = tf.summary.scalar(name+" Loss", model.normalized_cost)
            if mode == Mode.train:
                summ_lr = tf.summary.scalar("Learning Rate", model.config.learning_rate)
                summary = tf.summary.merge([summary, summ_lr])
            summ_layers = tf.summary.scalar("Layers", model.config.num_layers)
            summ_bs = tf.summary.scalar("Batch Size", model.config.batch_size)
            summ_grad_norm = tf.summary.scalar("Gradient Norm", model.config.max_grad_norm)
            summ_keep_prob = tf.summary.scalar("Dropout Keep Probability", model.config.keep_prob)
            summ_hidden_size = tf.summary.scalar("State Size", model.config.hidden_size)
            summary = tf.summary.merge([summary, summ_layers, summ_bs, summ_grad_norm, summ_keep_prob, summ_hidden_size])

    return model, summary


def search_params(base_config, train_set, validation_set):
    """
    Performs grid search over the value ranges in the reference configuration.
    :param base_config: Reference configuration with ranges as parameter values.
    :param train_set: Dataset to use for training.
    :param validation_set: Dataset to use for validation.
    :return: Configuration with smallest validation loss.
    """
    # Generate configs to be tested
    params = base_config.parameters
    configs = [Config()]

    for param in params:
        param_content = base_config.parameters[param]
        if isinstance(param_content, collections.Iterable):  # Check if the parameter is to be varied
            nr_configs = len(configs)
            for configID in range(nr_configs):
                for value in param_content:
                    new_config = copy.deepcopy(configs[0])
                    new_config.parameters[param] = value
                    configs.append(new_config)
                configs.pop(0)
        else:
            for config in configs:
                config.parameters[param] = base_config.parameters[param]

    log("Generated %d parameter configurations:" % len(configs))
    for i in range(len(configs)):
        log(str(i)+" "+configs[i].to_string())

    # Train and validate configs
    best_config = None
    best_loss = float("inf")

    counter = 0
    for config in configs:
        counter += 1
        log("############################### Running Configuration %d of %d ###############################" % (counter, len(configs)))

        validation_config = copy.deepcopy(config)
        validation_config.nr_epochs = 1

        with tf.Graph().as_default():
            mtrain, train_summary_op = instantiate_model(
                mode=Mode.train,
                config=config,
                data_set=train_set)

            mvalid, val_summary_op = instantiate_model(
                mode=Mode.validate,
                config=validation_config,
                data_set=validation_set)

            with tf.train.MonitoredTrainingSession() as session:
                run_loop(session, [Mode.train], [mtrain], [train_summary_op], logging=True, name_modifier=str(counter))
                config_loss = run_loop(session, [Mode.validate], [mvalid], [val_summary_op], logging=True, name_modifier=str(counter))[0][0]

        if best_loss > config_loss:
            best_config = config
            best_loss = config_loss
            log("Best loss: %.4f" % best_loss)
            log("Best config:"+best_config.to_string())

    log("Best total loss: %.4f" % best_loss)
    log("Best total config: "+best_config.to_string())

    return best_config


def make_iteration(config, train_set, validation_set, counter):
    validation_config = copy.deepcopy(config)
    validation_config.nr_epochs = 1

    with tf.Graph().as_default():
        mtrain, train_summary_op = instantiate_model(
            mode=Mode.train,
            config=config,
            data_set=train_set)

        mvalid, val_summary_op = instantiate_model(
            mode=Mode.validate,
            config=validation_config,
            data_set=validation_set)

        with tf.train.MonitoredTrainingSession() as session:
            run_loop(session, [Mode.train], [mtrain], [train_summary_op], logging=True, name_modifier=str(counter))
            config_loss = run_loop(session, [Mode.validate], [mvalid], [val_summary_op], logging=True, name_modifier=str(counter))[0][0]

    return config_loss


def search_params_parallel(base_config, train_set, validation_set):
    """
    Performs grid search over the value ranges in the reference configuration.
    :param base_config: Reference configuration with ranges as parameter values.
    :param train_set: Dataset to use for training.
    :param validation_set: Dataset to use for validation.
    :return: Configuration with smallest validation loss.
    """
    # Generate configs to be tested
    params = base_config.parameters
    configs = [Config()]

    for param in params:
        param_content = base_config.parameters[param]
        if isinstance(param_content, collections.Iterable):  # Check if the parameter is to be varied
            nr_configs = len(configs)
            for configID in range(nr_configs):
                for value in param_content:
                    new_config = copy.deepcopy(configs[0])
                    new_config.parameters[param] = value
                    configs.append(new_config)
                configs.pop(0)
        else:
            for config in configs:
                config.parameters[param] = base_config.parameters[param]

    log("Generated %d parameter configurations:" % len(configs))
    for i in range(len(configs)):
        log(str(i)+" "+configs[i].to_string())

    # Train and validate configs
    counter = 0

    pool = Pool()
    losses = []
    results = []
    for config in configs:
        results.append(pool.apply_async(make_iteration, [config, train_set, validation_set, counter]))
        counter += 1
    for result in results:
        losses.append(result.get())

    best_config_id = np.argmax(losses)
    best_loss = losses[best_config_id]
    best_config = configs[best_config_id]
    log("Best total loss: %.4f" % best_loss)
    log("Best total config: "+best_config.to_string())

    return best_config


def train_and_test(config, train_set, test_set, name_modifier=""):
    """
    Trains the model with the give configuration, tests it, and returns the model and the testing loss.
    :param config: Config to use for both training and testing.
    :param train_set: Training data
    :param test_set: Testing data
    :param name_modifier: Suffix to be appendet to the model's namestring (e.g. for tensorboard folder structure)
    :return: loss
    """
    with tf.Graph().as_default():
        mtrain, train_summary_op = instantiate_model(
            mode=Mode.train,
            config=config,
            data_set=train_set)

        mtest, test_summary_op = instantiate_model(
            mode=Mode.test,
            config=config,
            data_set=test_set)

        tf.train.get_or_create_global_step()
        saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=save_path, save_steps=10000)
        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(hooks=[saver_hook]) as session:
            run_loop(session, [Mode.train], [mtrain], [train_summary_op],
                     logging=True, name_modifier=name_modifier)
            loss = run_loop(session, [Mode.test], [mtest], [test_summary_op],
                            logging=True, name_modifier=name_modifier)

            log("Saving model to %s." % save_path)
            saver.save(session._sess._sess._sess._sess, save_path+os.sep+utils.start_time+'_final.ckpt')

    return loss


def run_loop(session, modes, models, summary_ops=None, logging=False, name_modifier=""):
    """
    Runs models for the specified number of epochs.
    :param session: MonitoredTrainingSession to use
    :param modes: List of modes. Whether to train, evaluate, or test
    :param models: List of models
    :param summary_ops: List of summary operations with one operation per model, or None.
    :param logging: Whether to write summaries for tensorboard.
    :param name_modifier: Suffix to identify the model in tensorboard. Creates a subdirectory.
    :return: List of lists, where each sub-list is the loss/epoch.
    """
    assert (len(modes) == len(models))
    assert (summary_ops is None or len(summary_ops) == len(models))
    assert (summary_ops is None or logging)

    nr_models = len(modes)
    representative_config = models[0].config  # used for variables that are assumed to be the same between models

    writers = []
    losses = []
    for i in range(nr_models):
        losses.append([])
        if logging:
            writers.append(
                tf.summary.FileWriter(log_path + os.sep + Mode.to_string(modes[i]).lower() + name_modifier,
                                      session.graph))
        else:
            writers.append(None)

    for i in range(representative_config.nr_epochs):
        for m in range(nr_models):
            model = models[m]
            perplexity = run_epoch(session, model,
                                   writer=writers[m],
                                   summary_op=summary_ops[m],
                                   eval_op=model.train_op if modes[m] == Mode.train else None)
            losses[m].append(perplexity)
            log("%s Epoch: %d Perplexity: %.3f" % (Mode.to_string(modes[m]), i + 1, perplexity))

    return losses


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

    search_nr_epochs = 1#20
    nr_epochs = 1#30

    search_config = GridSearchConfig()

    search_config.nr_epochs = search_nr_epochs

    config = search_params_parallel(search_config, train_set, validation_set)

    config.nr_epochs = nr_epochs

    train_and_test(config=config, train_set=train_set+validation_set, test_set=test_set, name_modifier="Final")


if __name__ == "__main__":
    tf.app.run()

# tensorboard --logdir=D:/Projekte/TFRegularizer/tfRegularizer/logs
# http://0.0.0.0:6006/
