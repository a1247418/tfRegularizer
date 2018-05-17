import tensorflow as tf
from enum import Enum


def data_type():
    return tf.float16 if tf.flags.FLAGS.use_fp16 else tf.float32


class ModelType(Enum):
    tf = 1
    interpolate = 2
    probabilistic = 3
    free_running = 4


class Mode(Enum):
    train = 1
    validate = 2
    test = 3

    def to_string(mode):
        if mode == Mode.train: string = "Train"
        elif mode == Mode.validate: string = "Validate"
        elif mode == Mode.test: string = "Test"
        else: raise ValueError(mode, 'is not a valid training mode.')
        return string


class SeqModel(object):
    def __init__(self, is_training, config, dataset):
        self._is_training = is_training
        self._dataset = dataset
        self._batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        self._config = config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        self._batch = tf.placeholder(tf.int32, shape=(None, None))  # Holds symbol IDs: [batch_size, num_steps]
        num_steps = tf.shape(self._batch)[1]

        # Embedding
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
              "embedding", [vocab_size, hidden_size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self._batch)

        # Dropout
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Build graph
        output, state = self._build_rnn_graph_lstm(inputs, config, is_training)

        # Output distribution
        # Softmax translates from hidden size to vocab size
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        # Add bias
        output = tf.reshape(output, [-1, hidden_size])
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self._batch_size, num_steps, vocab_size])

        # TODO: Here: introduce interpolation

        # batch should be [batch_size, num_steps]. It needs to be shifted by 1 for targets.
        targets = tf.concat([self._batch[:,1:], tf.zeros(shape=(tf.shape(self._batch)[0], 1), dtype=tf.int32)], axis=1)
        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([self._batch_size, num_steps], dtype=data_type()),  # weights
            average_across_timesteps=False,
            average_across_batch=False)
        # Masking - loss will be averaged over steps num_steps_in_batch outside of this function
        mask = tf.cast(tf.sign(tf.abs(targets)), tf.float32)  # Length of each seq
        masked_loss = tf.multiply(loss, mask)  # Multiply element-wise unwanted loss with 0
        sequence_loss = tf.reduce_sum(masked_loss, 1)  # Add up loss of sequences
        # sequence_loss /= tf.reduce_sum(mask, 1)  # Divide seq. loss by seq. lentgth
        total_loss = tf.reduce_sum(sequence_loss)  # avg over batch
        self._loss = masked_loss
        self._mask = mask
        self._target = targets
        self._logits = logits
        # Update the cost
        self._cost = total_loss
        self._final_state = state
        self._num_steps_in_batch = tf.reduce_sum(mask)
        # Cost normalized to batch - only used for logging. Otherwise normalization is done outside.
        self._normalized_cost = total_loss/self._num_steps_in_batch

        # Optimize, if is_training
        if is_training:
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                              config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)

    def _get_lstm_cell(self, config):
        # Forget bias set to 1
        return tf.contrib.rnn.BasicLSTMCell(
            config.hidden_size, forget_bias=1.0, state_is_tuple=True,
            reuse=tf.AUTO_REUSE)
        
    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
        def make_cell():
            cell = self._get_lstm_cell(config)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self._batch_size, data_type())

        outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    inputs,  # Tensor of shape: [batch_size, num_steps, ...]
                    initial_state=self._initial_state,
                    time_major=False)
        return outputs, state

    def assign_lr(self, session, lr_value):
        """Updates learning rate."""
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    """
    def export_ops(self, name):
        # Exports ops to collections.
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        # Imports ops from collections.
        if self._is_training:
          self._train_op = tf.get_collection_ref("train_op")[0]
          self._lr = tf.get_collection_ref("lr")[0]
          self._new_lr = tf.get_collection_ref("new_lr")[0]
          self._lr_update = tf.get_collection_ref("lr_update")[0]
          rnn_params = tf.get_collection_ref("rnn_params")
          if self._cell and rnn_params:
            params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                self._cell,
                self._cell.params_to_canonical,
                self._cell.canonical_to_params,
                rnn_params,
                base_variable_scope="Model/RNN")
            tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)
    """

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def batch(self):
        return self._batch

    @property
    def dataset(self):
        return self._dataset

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def cost(self):
        return self._cost

    @property
    def normalized_cost(self):
        return self._normalized_cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    """
    @property
    def input(self):
        return self._input
    """
    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

    @property
    def config(self):
        return self._config

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
        print("Preparing data set.")

        def data_generator():
            for element in data: yield element

        dataset = tf.data.Dataset().from_generator(data_generator, output_types=data_type())
        dataset = dataset.shuffle(buffer_size=10000, seed=64, reshuffle_each_iteration=True)
        # dataset = dataset.repeat(num_epochs)
        dataset = dataset.padded_batch(batch_size, padded_shapes=[None])
        dataset = dataset.prefetch(max(2, tf.contrib.data.AUTOTUNE))   # must be the last operaton of the pipeline

        print("Done preparing data set.")

        return dataset
