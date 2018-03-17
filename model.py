import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn

from data_utils import iobes_iob

class Model:
    def __init__(self, config):
        self.construct(config)
        
    def load(self, sess, path, load_vec, config, id_to_char, logger):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logger.info("Initialize model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
            if config["pre_emb"]:
                emb_weights = sess.run(self.char_lookup.read_value())
                emb_weights = load_vec(config["emb_file"], id_to_char, config["char_dim"], emb_weights)
                sess.run(self.char_lookup.assign(emb_weights))
                logger.info("Load pre-trained embedding.")
                
    def save(self, sess, path, model_name="ee.ckpt"):
        checkpoint_path = os.path.join(path, model_name)
        self.saver.save(sess, checkpoint_path)
        
    def construct(self, config):
        self.config = config
        # TODO: use a tensor type lr rate so learning rate could be dynamic
        self.lr = config['lr']
        
        self.char_dim = config['char_dim']
        self.lstm_dim = config['lstm_dim']
        self.seg_dim = config["seg_dim"]

        self.num_tags = config['num_tags']
        self.num_chars = config['num_chars']
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_score = tf.Variable(0.0, trainable=False)
        self.best_test_score = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        
        self.dataset = {} # a set of dataset
        self.iterator = tf.data.Iterator.from_structure((tf.int32, tf.int32, tf.int32), 
                (tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None])))
        
        # add inputs for the model
        self.char_inputs, self.seg_inputs, self.targets = self.iterator.get_next()
        
        # TODO: try not to use placeholder
        # dropout
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
        
        # lengths, batch_size, num_steps
        self.lengths = tf.cast(tf.reduce_sum(
            # 0: padding char, 1: used char; so the reduce sum should be lengths of the batch
            tf.sign(tf.abs(self.char_inputs)), reduction_indices=1), tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        
        # neural layers:
        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        # apply dropout before feed to lstm layer
        lstm_inputs = tf.layers.dropout(embedding, rate=self.dropout)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)
        lstm_outputs = self.biLSTM_layer(lstm_outputs, self.lstm_dim, self.lengths, name="2_lstm")

        # logits for tags
        self.logits = self.project_layer(lstm_outputs)
        
        # add predictal operation if not using crf
        if not config['crf']:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config['optimizer']
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config['clip'], self.config['clip']), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def set_dataset(self, data, dataset_name):
        # TODO
        def gen_data():
            for d in sorted(data, key=lambda x: len(x[0])):
                yield (d[1], d[2], d[3])
        self.dataset[dataset_name] = tf.data.Dataset.from_generator(gen_data, (tf.int32,tf.int32,tf.int32), 
                (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])))
    
    def make_dataset_init(self, dataset_name, batch_size=0, shuffle=0):
        '''
        TODO
        shuffle = 0 means dont shuffle
        '''
        if batch_size == 0:
            batch_size = self.config['batch_size']
        dataset_batch = self.dataset[dataset_name].padded_batch(batch_size, 
                padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])))
        dataset_batch = dataset_batch.shuffle(shuffle) if shuffle else dataset_batch
        return self.iterator.make_initializer(dataset_batch)
                
        
    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        # TODO: ??? when do we use pre-emb?
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [batch size, num_steps, embedding size(char_emb + seg_emb)], 
        """
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed
    
    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)
    
    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])
    
    
    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [batch_size, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            
            #            **logits**                                   **targets** 
            #             num_tags      1                                 1   
            #        |----------------|---|                           |--------|     
            #   1    |     small      | 0 |                           |num_tags|         
            #        |----------------|---|                           |--------|         
            #  num   |                |   |    *    batch_size,       | self.  |    *    batch_size 
            # steps  | project_logits |sma|                           | targets|             
            #        |                |ll |                           |        |      
            #        |----------------|---|                           |--------| 
            #
            # small = -1000
            # # pad logits for crf loss
            # start_logits = tf.concat(
            #     [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            # pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            # logits = tf.concat([project_logits, pad_logits], axis=-1)
            # logits = tf.concat([start_logits, logits], axis=1)
            # 
            # targets = tf.concat(
            #     [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
            # 
            # self.trans = tf.get_variable(
            #     "transitions",
            #     shape=[self.num_tags + 1, self.num_tags + 1],
            #     initializer=self.initializer)
            # log_likelihood, self.trans = crf_log_likelihood(
            #     inputs=logits,
            #     tag_indices=targets,
            #     transition_params=self.trans,
            #     sequence_lengths=lengths+1)
            # return tf.reduce_mean(-log_likelihood)

            if self.config['crf']:
                self.trans = tf.get_variable(
                    "transitions",
                    shape=[self.num_tags, self.num_tags],
                    initializer=self.initializer)
                log_likelihood, self.trans = crf_log_likelihood(
                    inputs=self.logits,
                    tag_indices=self.targets,
                    transition_params=self.trans,
                    sequence_lengths=lengths)
                return tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                mask = tf.sequence_mask(lengths+1)
                losses = tf.boolean_mask(losses, mask)
                return tf.reduce_mean(losses)
            
            
    def run_step(self, sess, is_train):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = {self.dropout: self.config['dropout']*is_train }
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            inputs, targets, lengths, logits = sess.run(
                [self.char_inputs, self.targets, self.lengths, self.logits], feed_dict)
            return inputs, targets, lengths, logits
        
    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        if self.config['crf']:
            #small = -1000.0
            #start = np.asarray([[small]*self.num_tags +[0]])
            for score, length in zip(logits, lengths):
                score = score[:length]
                logits = score
                #pad = small * np.ones([length, 1])
                #logits = np.concatenate([score, pad], axis=1)
                #logits = np.concatenate([start, logits], axis=0)
                path, _ = viterbi_decode(logits, matrix)
    
                #paths.append(path[1:])
                paths.append(path)
            return paths
        else:
            labels_pred = np.cast['int'](np.argmax(logits, axis=-1))
            return labels_pred
        
    def evaluate(self, sess, dataset_name, id_to_tag, id_to_char):
        sess.run(self.make_dataset_init(dataset_name))
        results = []
        trans = self.trans.eval()
        try:
            while True:
                inputs, tags, lengths, scores = self.run_step(sess, False)
                batch_paths = self.decode(scores, lengths, trans)
                for i in range(len(inputs)):
                    inp = inputs[i][:lengths[i]]
                    string = [id_to_char[ip] for ip in inp]
                    gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                    pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                    results.append([string, gold, pred])
                    
        except tf.errors.OutOfRangeError:
            pass
        
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)