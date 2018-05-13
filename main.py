"""
Based on the official tutorial:
https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/*.py
"""

import os
import time
import numpy as np
import tensorflow as tf
import dataReader
from model import SeqModel, ModelType
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
'''
flags.DEFINE_string("save_path", save_path,
                    "Where to save the model.")
flags.DEFINE_string("logs_path", logs_path,
                    "Where to save the (TF)logs.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("batch_size", 64,
                     "Batch size.")
'''
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

def data_type():
    return tf.float16 if tf.flags.FLAGS.use_fp16 else tf.float32


def get_time_str():
    return time.strftime("%d.%m. %H:%M:%S", time.gmtime())


class ModelInput(object):
    """Encapsulates parameters and data."""
    def __init__(self, config, data, name = None):
        self.batch_size = batch_size = config.batch_size
        self.dataset = ModelInput._prepare_data(data, batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

    @staticmethod
    def _prepare_data(data, batch_size):
        """Takes data array and returns a batched, 0-padded and shuffled dataset."""
        if PRINT_OUT: print("Preparing data set.")

        def data_generator():
            for element in data: yield element

        dataset = tf.data.Dataset().from_generator(data_generator, output_types=data_type())
        dataset = dataset.shuffle(buffer_size=10000, seed=64, reshuffle_each_iteration=True)
        # dataset = dataset.repeat(num_epochs)
        dataset = dataset.padded_batch(batch_size, padded_shapes=[None])
        dataset = dataset.prefetch(max(2, tf.contrib.data.AUTOTUNE))   # must be the last operaton of the pipeline

        if PRINT_OUT: print("Done preparing data set.")

        return dataset


class DefaultConfig(object):
    init_scale = 0.1
    learning_rate = 0.8
    max_grad_norm = 3
    num_layers = 1
    #num_steps = 10
    hidden_size = 2
    max_epoch = 5  # nr epochs with max learning rate
    max_max_epoch = 30
    keep_prob = 0.8
    lr_decay = 0.9
    batch_size = 64
    vocab_size = 29
    learning_mode = ModelType.tf


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


def main(_):
    train_set, validation_set, test_set, vocab, char2int = \
        dataReader.loadDataFromSaves() if DATA_FROM_SAVES else \
        dataReader.loadDataFromSource()
    config = DefaultConfig()
    config.vocab_size = len(vocab) + (0 not in vocab)  # Add padding token, if not already used

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = ModelInput(
                config=config, data=train_set, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                mtrain = SeqModel(
                    is_training=True, config=config, dataset=train_input)
                train_summ_loss = tf.summary.scalar("Training Loss", mtrain._normalized_cost)
                train_summ_lr = tf.summary.scalar("Learning Rate", mtrain.lr)
        
        with tf.name_scope("Valid"):
            valid_input = ModelInput(
                config=config, data=validation_set, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = SeqModel(
                    is_training=False, config=config, dataset=valid_input)
                val_summ_loss = tf.summary.scalar("Validation Loss", mvalid._normalized_cost)

        with tf.name_scope("Test"):
            test_input = ModelInput(
                config=config, data=test_set, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = SeqModel(
                    is_training=False, config=config, dataset=test_input)
                test_summ_loss = tf.summary.scalar("Test Loss", mtest._normalized_cost)

        # models = {"Train": mtrain, "Valid": mvalid, "Test": mtest}

        train_summary_op = tf.summary.merge([train_summ_lr, train_summ_loss])
        val_summary_op = tf.summary.merge([val_summ_loss])
        test_summary_op = tf.summary.merge([test_summ_loss])

        # Saver hook that allows the monitored session to automatically create checkpoints
        saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=save_path, save_steps=10000)
        #summary_hook = tf.train.SummarySaverHook(output_dir=logs_path, summary_op=train_summary_op, save_secs=10)

        #global_step = tf.Variable(0, trainable=False, name='global_step')

        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(hooks=[saver_hook]) as session:
            train_writer = tf.summary.FileWriter(logs_path+os.sep+"train", session.graph)
            val_writer = tf.summary.FileWriter(logs_path+os.sep+"val", session.graph)
            test_writer = tf.summary.FileWriter(logs_path+os.sep+"test", session.graph)

            # session = tf_debug.LocalCLIDebugWrapperSession(session)  # Enable debug

            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
                mtrain.assign_lr(session, config.learning_rate * lr_decay)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))

                train_perplexity = run_epoch(session, mtrain,
                                             writer=train_writer,
                                             summary_op=train_summary_op,
                                             eval_op=mtrain.train_op)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                valid_perplexity = run_epoch(session, mvalid,
                                             writer=val_writer,
                                             summary_op=val_summary_op)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest,
                                        writer=test_writer,
                                        summary_op=test_summary_op)
            print("Test Perplexity: %.3f" % test_perplexity)

            print("Saving model to %s." % save_path)
            saver.save(session, save_path, global_step=tf.train.global_step(session, tf.train.get_or_create_global_step()))

            # tensorboard --logdir=D:/Projekte/TFRegularizer/tfRegularizer/logs
            # http://0.0.0.0:6006/


if __name__ == "__main__":
    tf.app.run()
