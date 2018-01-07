import functools

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
"""
SummaRuNNer https://arxiv.org/pdf/1611.04230.pdf
With only content and salience term in the cost function.
Embedding trained with model instead of pretrain.
send model = word model in the original paper
paragraph model = sent model in the original paper
"""

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class HRNN:
    def __init__(self, input_x, input_y, output_dim,
                 num_word_hidden=100, num_sent_hidden=100, embedding_size=100,
                 Wc_size=100, Wr_size=(200, 100), vocabulary_size=100000,
                 num_class=2, learning_rate=0.002, reuse=False):
        # Placeholders for input, output and dropout
        self.input_x = input_x
        self.input_y = input_y
        self.output_dim = output_dim
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Model parameters
        self.num_word_hidden = num_word_hidden
        self.num_sent_hidden = num_sent_hidden
        self.embedding_size = embedding_size
        self.Wc_size = Wc_size
        self.Wr_size = Wr_size
        self.vocabulary_size = vocabulary_size
        self.num_class = num_class

        # Optimization parameters
        self.learning_rate = learning_rate
        self.global_step = None  # tf.Variable(0, name='global_step', trainable=False)
        self.is_training = tf.placeholder(tf.bool)  # placeholder for a single boolean value

        self.reuse = reuse

        # Init lazy properties
        self.forward_prop
        self.cost
        self.optimize

    @lazy_property
    def forward_prop(self):
        with tf.name_scope("embedding"):
            embeddings = tf.get_variable("word_embeddings", [self.vocabulary_size, self.embedding_size])
            word_embed = tf.nn.embedding_lookup(embeddings, self.input_x)
            print("embedding", word_embed.get_shape())

        with tf.name_scope("word_rnn"):
            gru_fw_cell = rnn.GRUCell(self.num_word_hidden, reuse=self.reuse)
            gru_bw_cell = rnn.GRUCell(self.num_word_hidden, reuse=self.reuse)
            (word_output_fw, word_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                gru_fw_cell, gru_bw_cell, word_embed, dtype=tf.float32, scope="word_rnn")
            print("word_output_fw", word_output_fw.get_shape())
            print("word_output_bw", word_output_bw.get_shape())

            word_stacked = tf.concat([word_output_fw, word_output_bw], axis=2)
            print("stacked", word_stacked.get_shape())
            word_averaged = tf.reduce_mean(word_stacked, axis=1)
            print("word_averaged", word_averaged.get_shape())

        with tf.name_scope("sent_rnn"):
            gru_fw_sent_cell = rnn.GRUCell(self.num_sent_hidden, reuse=self.reuse)
            gru_bw_sent_cell = rnn.GRUCell(self.num_sent_hidden, reuse=self.reuse)
            # [batch_size, max_time, input_size]
            print("expaned dim", tf.expand_dims(word_averaged, axis=0).get_shape())

            (sent_output_fw, sent_output_bw), sent_final_output = \
                tf.nn.bidirectional_dynamic_rnn(gru_fw_sent_cell,
                                                gru_bw_sent_cell,
                                                tf.expand_dims(word_averaged, axis=0),
                                                dtype=tf.float32, scope="sent_rnn")

            print("sent_output_fw", sent_output_fw.get_shape())
            print("sent_output_bw", sent_output_bw.get_shape())

            sent_stacked = tf.concat([sent_output_fw, sent_output_bw], axis=2)
            print("sent_stacked", sent_stacked.get_shape())
            sent_averaged = tf.reduce_mean(sent_stacked, axis=1)
            print("sent_averaged", sent_averaged.get_shape())

        with tf.name_scope("doc"):
            document_representation = tf.transpose(tf.layers.dense(inputs=sent_averaged, units=self.output_dim,
                                                                   activation=tf.nn.tanh, reuse=self.reuse))
            print("document_representation", document_representation.get_shape())

        with tf.name_scope("output_layer"):
            Wc = tf.get_variable("wc", shape=[1, self.Wc_size], dtype=tf.float32,
                                 initializer=tf.zeros_initializer)
            Wr = tf.get_variable("Wr", shape=list(self.Wr_size), dtype=tf.float32,
                                 initializer=tf.zeros_initializer)

            h_j = tf.tanh(tf.transpose(tf.squeeze(sent_stacked, axis=0)))

            content = tf.transpose(tf.matmul(Wc, h_j))
            print("content cost", content.get_shape())
            salience = tf.matmul(tf.matmul(tf.transpose(h_j), Wr), document_representation)
            print("salience cost", salience.get_shape())

            logits = content + salience

        self.reuse = True

        return logits

    @lazy_property
    def cost(self):
        labels = tf.cast(self.input_y, tf.float32)
        with tf.name_scope("cost"):
            b = tf.get_variable("b", shape=[1], dtype=tf.float32,
                                initializer=tf.zeros_initializer)
            logits = self.forward_prop + b
            p_y1 = tf.nn.sigmoid(logits)
            # clip value to avoid NAN
            log_p_y1 = tf.log(tf.clip_by_value(p_y1, 1e-10, 1.0))
            log_p_y0 = tf.log(tf.clip_by_value(1 - p_y1, 1e-10, 1.0))
            error = -tf.reduce_sum(tf.multiply(labels, log_p_y1) + tf.multiply((1 - labels), log_p_y0))
        correct_prediction = tf.equal(tf.argmax(input=tf.concat([1 - p_y1, p_y1], axis=1), axis=1),
                                      tf.squeeze(self.input_y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return error, p_y1, accuracy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer()
        error, p_y1, accuracy = self.cost
        gvs = optimizer.compute_gradients(error)
        # Gradient clipping by value
        capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
        return optimizer.apply_gradients(capped_gvs)

