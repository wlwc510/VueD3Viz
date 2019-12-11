# -*- coding: utf-8 -*
from tensorflow.contrib.rnn import DropoutWrapper
from utils import *
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope as vs


BATCH_SIZE = config.FLAGS.batch_size
unit_num = embeddings_size         # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
time_step = max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = config.FLAGS.dropout
EPOCH = config.FLAGS.epoch
TAGS_NUM = get_class_size()
SAVE_PREDICT = config.FLAGS.save_pred


class NER_net:
    def __init__(self, scope_name, iterator, embedding, batch_size, iterator_test=None):
        '''
        :param scope_name:
        :param iterator: 调用tensorflow DataSet API把数据feed进来。
        :param embedding: 提前训练好的word embedding
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.embedding = embedding
        self.iterator = iterator
        # if iterator_test != None:
        self.iterator_test = iterator_test
        with tf.variable_scope(scope_name) as scope:
            self._build_net()

    def _build_net(self):

        self.global_step = tf.Variable(0, trainable=False)
        source = self.iterator.source
        tgt = self.iterator.target_input
        if self.iterator_test != None:
            source_test = self.iterator_test.source
            tgt_test = self.iterator_test.target_input
        # 得到当前batch的长度（如果长度不足的会被padding填充）
        max_sequence_in_batch = self.iterator.source_sequence_length
        max_sequence_in_batch = tf.reduce_max(max_sequence_in_batch)
        max_sequence_in_batch = tf.to_int32(max_sequence_in_batch)

        # x: [batch_size, time_step, embedding_size], float32  time_step 为句子长度
        self.x = tf.nn.embedding_lookup(self.embedding, source)
        # y: [batch_size, time_step]
        self.y = tgt

        if self.iterator_test != None:
            self.x_test = tf.nn.embedding_lookup(self.embedding, source_test) # (6, 64, 300)
            self.y_test = tgt_test # (6, 64)

        cell_forward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        if DROPOUT_RATE is not None:
            cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)
            cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)

        # time_major 可以适应输入维度。
        outputs, bi_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x, dtype=tf.float32)

        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)

        # projection:
        W = tf.get_variable("projection_w", [2 * unit_num, TAGS_NUM])
        b = tf.get_variable("projection_b", [TAGS_NUM])
        x_reshape = tf.reshape(outputs, [-1, 2 * unit_num])
        projection = tf.matmul(x_reshape, W) + b

        # -1 to time step
        self.outputs = tf.reshape(projection, [self.batch_size, -1, TAGS_NUM])

        self.seq_length = tf.convert_to_tensor(self.batch_size * [max_sequence_in_batch], dtype=tf.int32)
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, self.seq_length)

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        ## test
        if self.iterator_test != None:
            # time_major 可以适应输入维度。
            outputs_test, bi_state_test = \
                tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x_test, dtype=tf.float32)

            forward_out_test, backward_out_test = outputs_test
            outputs_test = tf.concat([forward_out_test, backward_out_test], axis=2)

        # projection:

            self.x_reshape_test = tf.reshape(outputs_test, [-1, 2 * unit_num]) # (384,600)
            self.projection_test = tf.matmul(self.x_reshape_test, W) + b # (384,16)

            self.outputs_test = tf.reshape(self.projection_test, [100, -1, TAGS_NUM]) # (6, 64, 16)
            self.seq_length_test = tf.convert_to_tensor(100 * [max_sequence_in_batch], dtype=tf.int32)

            num_tags = tensor_shape.dimension_value(self.outputs_test.shape[2]) # 16

            self.log_likelihood_test, self.transition_params_test = tf.contrib.crf.crf_log_likelihood(
                self.outputs_test, self.y_test, self.seq_length_test, vs.get_variable("transitions_test", [num_tags, num_tags]))

            # Add a training op to tune the parameters.
            self.loss_test = tf.reduce_mean(-self.log_likelihood_test)
            # self.train_op_test = tf.train.AdamOptimizer().minimize(self.loss_test)

def train(net, iterator, iterator_test, sess):
    saver = tf.train.Saver(max_to_keep=5)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        print('loading pre-trained model from %s.....' % path)
        saver.restore(sess, path)

    current_epoch = sess.run(net.global_step)
    # trainlist_file = r'W:\Rs project data\NER-损失误差\NER-master\resource\train_loss.txt'
    # testlist_file = r'W:\Rs project data\NER-损失误差\NER-master\resource\test_loss.txt'
    # trainlist = []
    # testlist = []
    loss1 = []
    while True:
        print(current_epoch)
        if current_epoch > EPOCH: break
        try:
            DROPOUT_RATE = 0.8
            tf_unary_scores, tf_transition_params, _, losses = sess.run(
                [net.outputs, net.transition_params, net.train_op, net.loss])

            DROPOUT_RATE = 1.0
            tf_unary_scores_test, tf_transition_params_test, losses_test = sess.run(
                [net.outputs_test, net.transition_params_test, net.loss_test])
            loss1.append(losses_test)
            loss2 = losses_test

            if loss2 <= min(loss1) * 1.2: # [1, 1.2 , 1.5] 分别对应 model model1_2 model1_5
                print('*' * 100)
                print(current_epoch, 'loss', losses)
                print(current_epoch, 'test_loss', losses_test)
                sess.run(tf.assign(net.global_step, current_epoch))
                saver.save(sess, model_path + 'points', global_step=current_epoch)
            # if current_epoch % 50 == 0:
            #
            #     with open(trainlist_file, 'a', encoding='utf-8') as f1:
            #         f1.write(str(losses) + '\n')
            #     with open(testlist_file, 'a', encoding='utf-8') as f2:
            #         f2.write(str(losses_test) + '\n')

            # # 每隔10%的进度则save一次。
            # if current_epoch % (EPOCH / 10) == 0 and current_epoch != 0:
            #     sess.run(tf.assign(net.global_step, current_epoch))
            #     saver.save(sess, model_path+'points', global_step=current_epoch)

            current_epoch += 1

        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
            sess.run(iterator_test.initializer)
        except tf.errors.InvalidArgumentError:
            # iterator.next() cannot get enough data to a batch, initialize it.
            # 正常初始化流程
            sess.run(iterator.initializer)
            sess.run(iterator_test.initializer)

    print('training finished! test error')

# def train(net, iterator, sess):
#     saver = tf.train.Saver()
#     ckpt = tf.train.get_checkpoint_state(model_path)
#     if ckpt is not None:
#         path = ckpt.model_checkpoint_path
#         print('loading pre-trained model from %s.....' % path)
#         saver.restore(sess, path)
#
#     current_epoch = sess.run(net.global_step)
#     while True:
#         if current_epoch > EPOCH: break
#         try:
#             tf_unary_scores, tf_transition_params, _, losses = sess.run(
#                 [net.outputs, net.transition_params, net.train_op, net.loss])
#
#             if current_epoch % 100 == 0:
#                 print('*' * 100)
#                 print(current_epoch, 'loss', losses)
#                 print('*' * 100)
#
#             # 每隔10%的进度则save一次。
#             if current_epoch % (EPOCH / 10) == 0 and current_epoch != 0:
#                 sess.run(tf.assign(net.global_step, current_epoch))
#                 saver.save(sess, model_path+'points', global_step=current_epoch)
#
#             current_epoch += 1
#
#         except tf.errors.OutOfRangeError:
#             sess.run(iterator.initializer)
#         except tf.errors.InvalidArgumentError:
#             # iterator.next() cannot get enough data to a batch, initialize it.
#             # 正常初始化流程
#             sess.run(iterator.initializer)
#     print('training finished!')


def predict(net, tag_table, sess):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        print('loading pre-trained model from %s.....' % path)
        saver.restore(sess, path)
    else:
        print('Model not found, please train your model first')
        return

    # 获取原文本的iterator
    file_iter = file_content_iterator(pred_file)

    while True:
        try:
            tf_unary_scores, tf_transition_params = sess.run(
                [net.outputs, net.transition_params])
        except tf.errors.OutOfRangeError:
            print('Prediction finished!')
            break

        # 把batch那个维度去掉
        tf_unary_scores = np.squeeze(tf_unary_scores)

        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores, tf_transition_params)
        tags = []
        for id in viterbi_sequence:
            tags.append(sess.run(tag_table.lookup(tf.constant(id, dtype=tf.int64))))
        # tag = ' '.join('%s' %id for id in tags)
        tag = ' '.join(list(idx.decode('utf-8') for idx in tags))
        # with open(SAVE_PREDICT, 'a', encoding='utf-8') as f1:
        #     f1.write(tag + '\n')
        # write_result_to_file(file_iter, tags)


if __name__ == '__main__':

    action = config.FLAGS.action
    # 获取词的总数。
    vocab_size = get_src_vocab_size() #

    src_unknown_id = tgt_unknown_id = vocab_size
    src_padding = vocab_size + 1

    src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id,
                                                           tgt_unknown_id)


    embedding = load_word2vec_embedding(vocab_size)

    if action == 'train':
        iterator = get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, BATCH_SIZE)
        # DROPOUT_RATE = 1.0
        iterator_test = get_test_iterator(src_vocab_table, tgt_vocab_table, vocab_size, 100)
    elif action == 'predict':
        BATCH_SIZE = 1
        DROPOUT_RATE = 1.0
        iterator = get_predict_iterator(src_vocab_table, vocab_size, BATCH_SIZE)
    # else:
    #     print('Only support train and predict actions.')
    #     exit(0)

    tag_table = tag_to_id_table()
    if action == 'train':
        net = NER_net("ner", iterator, embedding, BATCH_SIZE, iterator_test)
    elif action == 'predict':
        net = NER_net("ner", iterator, embedding, 1)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        if action == 'train':
            sess.run(iterator_test.initializer)

        # iterator = sess.run(iterator)
        # iterator_test = sess.run(iterator_test)
        # net = NER_net("ner", iterator, embedding, BATCH_SIZE, iterator_test)

        '''
  
        a = sess.run(net.y_test)
        print(a)
        print(a.shape)
        '''

        # x, y, seq, tf_transition_params, _, losses, pro, x_reshape, outputs_test, log_likelihood_test = sess.run(
        #     [net.x_test, net.y_test, net.seq_length_test, net.transition_params, net.train_op, net.loss, net.projection_test, net.x_reshape_test, net.outputs_test,
        #      net.log_likelihood_test])
        #
        # print(x.shape)
        # print(y.shape)
        # print(seq.shape)
        # # print(tf_unary_scores.shape)
        # print(tf_transition_params.shape)
        # print(pro.shape)
        # print(x_reshape.shape)
        # print(outputs_test.shape)
        # print(log_likelihood_test.shape)

        if action == 'train':
            train(net, iterator, iterator_test, sess)
        elif action == 'predict':
            predict(net, tag_table, sess)