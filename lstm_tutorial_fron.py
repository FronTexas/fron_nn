import tensorflow as tf
import numpy as np


class Input:
    def __init__(self, data, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)


class Model:
    def __init__(self, input_obj, is_training, num_layers, hidden_layers_size, vocab_size,
                 dropout=0.5, init_scale=0.05):

        self.input_obj = input_obj
        self.is_training = is_training
        self.num_layers = num_layers
        self.hidden_layers_size = hidden_layers_size
        self.vocab_size = vocab_size

        # Create the embedding for input
        with tf.device("/cpu:0"):
            embeddings = tf.Variable(
                tf.random.uniform(shape=[vocab_size, hidden_layers_size], minval=-init_scale, maxval=init_scale))
            inputs = tf.nn.embedding_lookup(params=embeddings, ids=input_obj.input_data)

        if dropout < 1 and is_training:
            inputs = tf.nn.dropout(x=inputs, keep_prob=dropout)

        self.states = tf.placeholder(dtype=tf.dtypes.float32,
                                     shape=[num_layers, 2, input_obj.batch_size, hidden_layers_size],
                                     name='states')

        state_per_layer = tf.unstack(value=self.states, axis=0)
        state_tuples = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(c=state[0], h=state[1])
             for state in state_per_layer]
        )

        cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_layers_size, state_is_tuple=True)
        if dropout < 1 and is_training:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=dropout)

        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[
                cell for _ in num_layers
            ], state_is_tuple=True)

        output, self.current_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=state_tuples)
        output = tf.reshape(output, shape=[-1, hidden_layers_size])  # shape (20*35, hidden_layers_size)

        softmax_w = tf.Variable(
            tf.random.uniform(shape=[hidden_layers_size, vocab_size], minval=init_scale, maxval=init_scale))
        softmax_b = tf.Variable(tf.random.uniform(shape=[vocab_size]))
        logits = tf.nn.xw_plus_b(x=output, weights=softmax_w, biases=softmax_b)
        logits = tf.reshape(tensor=logits, shape=[input_obj.batch_size, input_obj.num_steps, vocab_size])

        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=input_obj.targets,
            weights=tf.ones(shape=input_obj.targets.shape)
        )

        logits = tf.reshape(tensor=logits, shape=[-1, vocab_size])
        self.softmax_logits = tf.nn.softmax(logits=logits, axis=1)
        self.prediction = tf.cast(x=tf.argmax(input=softmax_logits, axis=1), dtype=tf.dtypes.int32)
        targets = tf.reshape(input_obj.targets, shape=[-1])

        # TODO understand what tf.equal returns
        # TODO Wrap tf.equal with tf.reduce_mean
        self.accuracy = tf.equal(x=self.prediction, y=targets)

        # TODO add if not is_training: return

        # TODO initialize learning_rate to be variable(0.0)

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(t_list=tf.gradients(ys=self.cost, xs=trainable_variables), clip_norm=5)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            grads_and_vars=zip(grads, trainable_variables),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )

        # TODO create a lr_update op that updates the learning_rate


def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y


def read_words(filename):
    pass


def load_data():
    words = read_words()
    pass


def train(data, num_epochs, batch_size=20, num_steps=35, num_layers=2):
    training_data = Input(data, batch_size, num_steps)
    m = Model(input_data=training_data, num_layers=num_layers)
    for epoch in num_epochs:
        X_batches, Y_batches = create_batches()

        for current_x_batch, current_y_batch in zip(X_batches, Y_batches):
            with tf.session() as sess:
                loss, accuracy = sess.run([m.loss, m.accuracy], feed_dict={x: current_x_batch, y: current_y_batch})


def test():
    pass


def main():
    is_training = True
    if is_training:
        train()
    else:
        test()
