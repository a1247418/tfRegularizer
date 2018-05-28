import copy
import collections
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
from model import SeqModel, ModelType, ModelInput, Mode
from configurations import Config, DefaultConfig, GridSearchConfig
from utils import log
import trainingControl as tCtrl


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
            mtrain, train_summary_op = tCtrl.instantiate_model(
                mode=Mode.train,
                config=config,
                data_set=train_set)

            mvalid, val_summary_op = tCtrl.instantiate_model(
                mode=Mode.validate,
                config=validation_config,
                data_set=validation_set)

            with tf.train.MonitoredTrainingSession() as session:
                tCtrl.run_loop(session, [Mode.train], [mtrain], [train_summary_op], logging=True, name_modifier=str(counter))
                config_loss = tCtrl.run_loop(session, [Mode.validate], [mvalid], [val_summary_op], logging=True, name_modifier=str(counter))[0][0]

        if best_loss > config_loss:
            best_config = config
            best_loss = config_loss
            log("Best loss: %.4f" % best_loss)
            log("Best config:"+best_config.to_string())

    log("Best total loss: %.4f" % best_loss)
    log("Best total config: "+best_config.to_string())

    return best_config


def make_iteration(config, train_set, validation_set, counter):
    """
    Evaluate the model built with a specific parameter configuration.
    :param config: Configuration to use
    :param train_set: Data to train on
    :param validation_set: Data to validate on
    :param counter: Index of the configuration. This is solely used to create a logging sub-directory.
    :return: validation loss
    """
    validation_config = copy.deepcopy(config)
    validation_config.nr_epochs = 1

    with tf.Graph().as_default():
        mtrain, train_summary_op = tCtrl.instantiate_model(
            mode=Mode.train,
            config=config,
            data_set=train_set)

        mvalid, val_summary_op = tCtrl.instantiate_model(
            mode=Mode.validate,
            config=validation_config,
            data_set=validation_set)

        with tf.train.MonitoredTrainingSession() as session:
            tCtrl.run_loop(session, [Mode.train], [mtrain], [train_summary_op], logging=True, name_modifier=str(counter))
            config_loss = tCtrl.run_loop(session, [Mode.validate], [mvalid], [val_summary_op], logging=True, name_modifier=str(counter))[0][0]

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

    nr_threads = tf.flags.FLAGS.nr_threads
    log("Running parameter search with %d threads" % nr_threads)
    pool = Pool(processes=nr_threads)
    losses = []
    results = []

    for cohort_id in range(nr_configs//nr_threads + 1):
        for process_id in range(nr_configs):
            total_id = process_id + cohort_id * nr_threads
            if total_id >= nr_configs: break
            results.append(pool.apply_async(make_iteration, [configs[total_id], train_set, validation_set, counter]))
            counter += 1

        for process_id in range(nr_configs):
            total_id = process_id + cohort_id * nr_threads
            if total_id >= nr_configs: break
            losses.append(results[total_id].get())

    best_config_id = np.argmax(losses)
    best_loss = losses[best_config_id]
    best_config = configs[best_config_id]
    log("Best total loss: %.4f" % best_loss)
    log("Best total config: "+best_config.to_string())

    return best_config
