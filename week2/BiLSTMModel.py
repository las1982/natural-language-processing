import tensorflow as tf
import numpy as np


class BiLSTMModel:
    def __init__(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
        self.declare_placeholders()
        self.build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
        self.compute_predictions()
        self.compute_loss(n_tags, PAD_index)
        self.perform_optimization()

    def declare_placeholders(self):
        """Specifies placeholders for the model."""

        # Placeholders for input and ground truth output.
        self.input_batch = tf.placeholder(tf.int32, [None, None], 'input_batch')
        self.ground_truth_tags = tf.placeholder(tf.int32, [None, None], 'ground_truth_tags')

        # Placeholder for lengths of the sequences.
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')

        # Placeholder for a dropout keep probability. If we don't feed
        # a value for this placeholder, it will be equal to 1.0.
        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=(), name="dropout_ph")

        # Placeholder for a learning rate (tf.float32).
        self.learning_rate_ph = tf.placeholder(tf.float32, name="learning_rate_ph")

    def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
        """Specifies bi-LSTM architecture and computes logits for inputs."""

        # Create embedding variable (tf.Variable) with dtype tf.float32
        initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
        embedding_matrix_variable = tf.Variable(initial_embedding_matrix, dtype=tf.float32, name="embeddings_matrix")

        # Create RNN cells (for example, tf.nn.rnn_cell.BasicLSTMCell) with n_hidden_rnn number of units
        # and dropout (tf.nn.rnn_cell.DropoutWrapper), initializing all *_keep_prob with dropout placeholder.
        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_rnn)
        forward_cell = tf.nn.rnn_cell.DropoutWrapper(forward_cell, self.dropout_ph, self.dropout_ph, self.dropout_ph)
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_rnn)
        backward_cell = tf.nn.rnn_cell.DropoutWrapper(backward_cell, self.dropout_ph, self.dropout_ph, self.dropout_ph)

        # Look up embeddings for self.input_batch (tf.nn.embedding_lookup).
        # Shape: [batch_size, sequence_len, embedding_dim].
        embeddings = tf.nn.embedding_lookup(params=embedding_matrix_variable, ids=self.input_batch)

        # Pass them through Bidirectional Dynamic RNN (tf.nn.bidirectional_dynamic_rnn).
        # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn].
        # Also don't forget to initialize sequence_length as self.lengths and dtype as tf.float32.
        (rnn_output_fw, rnn_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, embeddings,
                                                                            sequence_length=self.lengths,
                                                                            dtype=tf.float32)
        rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

        # Dense layer on top.
        # Shape: [batch_size, sequence_len, n_tags].
        self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)

    def compute_predictions(self):
        """Transforms logits to probabilities and finds the most probable tags."""

        # Create softmax (tf.nn.softmax) function
        softmax_output = tf.nn.softmax(self.logits)

        # Use argmax (tf.argmax) to get the most probable tags
        # Don't forget to set axis=-1
        # otherwise argmax will be calculated in a wrong way
        self.predictions = tf.argmax(softmax_output, axis=-1)

    def compute_loss(self, n_tags, PAD_index):
        """Computes masked cross-entopy loss with logits."""

        # Create cross entropy function function (tf.nn.softmax_cross_entropy_with_logits)
        ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth_tags_one_hot, logits=self.logits)

        mask = tf.cast(tf.not_equal(self.input_batch, PAD_index), tf.float32)
        # Create loss function which doesn't operate with <PAD> tokens (tf.reduce_mean)
        # Be careful that the argument of tf.reduce_mean should be
        # multiplication of mask and loss_tensor.
        self.loss = tf.reduce_mean(tf.multiply(mask, loss_tensor))

    def perform_optimization(self):
        """Specifies the optimizer and train_op for the model."""

        # Create an optimizer (tf.train.AdamOptimizer)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # Gradient clipping (tf.clip_by_norm) for self.grads_and_vars
        # Pay attention that you need to apply this operation only for gradients
        # because self.grads_and_vars contains also variables.
        # list comprehension might be useful in this case.
        clip_norm = tf.cast(1.0, tf.float32)
        #     map_fn(lambda x: (x[0], x[1]), self.grads_and_vars)
        self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for (grad, var) in self.grads_and_vars]

        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
        print()

    def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability):
        feed_dict = {self.input_batch: x_batch,
                     self.ground_truth_tags: y_batch,
                     self.learning_rate_ph: learning_rate,
                     self.dropout_ph: dropout_keep_probability,
                     self.lengths: lengths}

        session.run(self.train_op, feed_dict=feed_dict)

    def predict_for_batch(self, session, x_batch, lengths):
        ######################################
        ######### YOUR CODE HERE #############
        feed_dict = {self.input_batch: x_batch,
                     self.lengths: lengths}
        ######################################
        predictions = session.run(self.predictions, feed_dict=feed_dict)
        return predictions
