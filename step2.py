# encoding: utf-8

from sklearn.model_selection import train_test_split
import tensorflow as tf
print('\ntensorflow: {}\ntf.test.is_gpu_available: {}\n'.format(tf.__version__, tf.test.is_gpu_available()))
import numpy as np
import sys, os, time, codecs, pdb

sys.path.append('./utils')
from tf_funcs import *
from pre_data_bert import *


FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', './data/ECF_glove_300.txt', 'embedding file')
tf.app.flags.DEFINE_string('path', './data/', 'path for dataset')
tf.app.flags.DEFINE_string('video_emb_file', './data/video_embedding_3dcnn_4096.npy', 'ndarray (13620, 4096)')
tf.app.flags.DEFINE_string('audio_emb_file', './data/audio_embedding_6373.npy', 'ndarray (13620, 6373)')
tf.app.flags.DEFINE_string('video_idx_file', './data/video_id_mapping.npy', 'mapping dict: {dia1utt1: 1, ...}')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 35, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('pred_future_cause', 1, 'whether consider the cause among future utterances')
## model struct ##
tf.app.flags.DEFINE_string('choose_emocate', '', 'whether choose emocate')
tf.app.flags.DEFINE_integer('emocate_eval', 6, 'Whether to save the best evaluation score of w-avg 6 or w-avg 4')
tf.app.flags.DEFINE_string('use_x_v', '', 'whether use video embedding')
tf.app.flags.DEFINE_string('use_x_a', '', 'whether use audio embedding')
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('batch_size', 200, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')
tf.app.flags.DEFINE_float('keep_prob_v', 0.5, 'training dropout keep prob for visual features')
tf.app.flags.DEFINE_float('keep_prob_a', 0.5, 'training dropout keep prob for audio features')
tf.app.flags.DEFINE_integer('end_run', 21, 'end_run')
tf.app.flags.DEFINE_integer('training_iter', 12, 'number of train iter')

tf.app.flags.DEFINE_string('log_path', './log', '')
tf.app.flags.DEFINE_string('scope', 'TEMP', 'scope')
tf.app.flags.DEFINE_string('log_file_name', 'step2.log', 'name of log file')
tf.app.flags.DEFINE_string('save_pair', 'yes', 'whether save predicted pairs')
tf.app.flags.DEFINE_string('step1_file_dir', 'step1/', 'file directory of step1')


def print_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    print('choose_emocate: {}\nemocate_eval: {}\nvideo_emb_file: {}\naudio_emb_file: {}\nuse_x_v: {}\nuse_x_a: {}\n\n'.format(
        FLAGS.choose_emocate, FLAGS.emocate_eval, FLAGS.video_emb_file, FLAGS.audio_emb_file, FLAGS.use_x_v,  FLAGS.use_x_a))

    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print('path: {}\nbatch: {}\nlr: {}\nkb1: {}\nkb2: {}\nl2_reg: {}\nkeep_prob_v: {}\nkeep_prob_a: {}\ntraining_iter: {}\nend_run: {}\npred_future_cause: {}\nstep1_file_dir: {}\n\n'.format(
        FLAGS.path, FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg, FLAGS.keep_prob_v, FLAGS.keep_prob_a, FLAGS.training_iter, FLAGS.end_run, FLAGS.pred_future_cause, FLAGS.step1_file_dir))


def build_model(embeddings, placeholders, RNN=biLSTM):
    word_embedding, pos_embedding, video_embedding, audio_embedding = embeddings
    x, sen_len, distance, x_emocate, x_v, y, is_training = placeholders

    sen_len = tf.reshape(sen_len, [-1])
    inputs = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(inputs, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    inputs = tf.nn.dropout(inputs, keep_prob = is_training * FLAGS.keep_prob1 + (1.-is_training))
    dis = tf.nn.embedding_lookup(pos_embedding, distance) # 66-134
    x_emocate = tf.nn.embedding_lookup(pos_embedding, x_emocate) # 1-6
    
    x_a = tf.nn.embedding_lookup(audio_embedding, x_v)
    x_v = tf.nn.embedding_lookup(video_embedding, x_v)
    x_v = tf.nn.dropout(x_v, keep_prob = is_training * FLAGS.keep_prob_v + (1.-is_training))
    x_a = tf.nn.dropout(x_a, keep_prob = is_training * FLAGS.keep_prob_a + (1.-is_training))

    h2 = 2 * FLAGS.n_hidden
    if FLAGS.use_x_v:
        x_v = tf.nn.relu(layer_normalize(tf.layers.dense(x_v, h2, use_bias=True)))

    if FLAGS.use_x_a:
        x_a = tf.nn.relu(layer_normalize(tf.layers.dense(x_a, h2, use_bias=True)))
        

    def concate_feature(s1, x_v, x_a):
        if FLAGS.use_x_v:
            s1 = tf.concat([s1, x_v], axis = 2)
        if FLAGS.use_x_a:
            s1 = tf.concat([s1, x_a], axis = 2)
        return s1
    
    def get_s(inputs, sen_len, name):
        with tf.name_scope('word_encode'):  
            inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer' + name)
        with tf.name_scope('word_attention'):
            w1 = get_weight_varible('word_att_w1' + name, [h2, h2])
            b1 = get_weight_varible('word_att_b1' + name, [h2])
            w2 = get_weight_varible('word_att_w2' + name, [h2, 1])
            s = att_var(inputs,sen_len,w1,b1,w2)
        s = tf.reshape(s, [-1, 2, h2])
        return s
    
    
    s = get_s(inputs, sen_len, name='_word_encode') # [-1, 2 * h2]
    s = concate_feature(s, x_v, x_a)

    dim_s = s.shape[-1].value
    s = tf.reshape(s, [-1, 2 * dim_s])
    if FLAGS.choose_emocate:
        s = tf.concat([s, dis, x_emocate], 1)
    else:
        s = tf.concat([s, dis], 1)

    s1 = tf.nn.dropout(s, keep_prob=is_training * FLAGS.keep_prob2 + (1.-is_training))
    n_hidden = s.shape[-1].value
    w_pair = get_weight_varible('softmax_w_pair', [n_hidden, FLAGS.n_class])
    b_pair = get_weight_varible('softmax_b_pair', [FLAGS.n_class])
    pred_pair = tf.nn.softmax(tf.matmul(s1, w_pair) + b_pair)
        
    reg = tf.nn.l2_loss(w_pair) + tf.nn.l2_loss(b_pair)
    return pred_pair, reg


class Dataset(object):
    def __init__(self, data_file_name, word_idx, video_idx):
        x, sen_len, distance, x_emocate, x_v, y, pair_id_all, pair_id, doc_id_list, y_pairs = load_data_utt_step2(data_file_name, word_idx, video_idx, FLAGS.max_sen_len, FLAGS.choose_emocate, FLAGS.pred_future_cause)

        self.x, self.sen_len, self.distance, self.x_emocate, self.x_v, self.y, self.pair_id_all, self.pair_id, self.doc_id_list, self.y_pairs = x, sen_len, distance, x_emocate, x_v, y, pair_id_all, pair_id, doc_id_list, y_pairs
        self.all = [x, sen_len, distance, x_emocate, x_v, y]

def get_batch_data(dataset, is_training, batch_size):
    test = bool(1 - is_training)
    for index in batch_index(len(dataset.x), batch_size, test):
        feed_list = list(map(lambda x: x[index], dataset.all)) + [is_training]
        yield feed_list, index


from collections import defaultdict
def create_dict(pair_list, choose_emocate):
    emotion_idx_rev = dict(zip(range(7), ['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']))
    pair_dict = defaultdict(list)  # {dia_id: [emo_id,cause_id]}
    for x in pair_list:
        if choose_emocate:
            tmp = x[1:3] + [emotion_idx_rev[x[3]]]
            pair_dict[x[0]].append(tmp)
        else:
            pair_dict[x[0]].append(x[1:-1])
    return pair_dict

def write_data_all(input_file_name, output_file_name, dataset, pred_y, tr_batch_index, choose_emocate=''): # tr_batch_index存训练集打乱的索引
    # pair_id_all: [[doc_id, p[0], p[1], y_emo_tmp[p[0]-1]]]
    # pair_id: [[doc_id, i, j, predy_emo_tmp[i-1]]] / [[doc_id, i, j, y_emo_tmp[i-1]]]
    pair_id_all, pair_id, doc_id_list, y_pairs = dataset.pair_id_all, dataset.pair_id, dataset.doc_id_list, dataset.y_pairs
    if tr_batch_index:
        pair_id = np.array(pair_id)[tr_batch_index]
    
    print('pair_id: {}  pred_y: {}'.format(len(pair_id), len(pred_y))) # 训练集最后凑不满一个batch的数据会舍掉
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_tmp = list(pair_id[i])
            pair_id_filtered.append(pair_id_tmp)

    pair_id_all_dict = create_dict(pair_id_all, choose_emocate)
    pair_id_filtered_dict = create_dict(pair_id_filtered, choose_emocate)

    fo = open(output_file_name, 'w', encoding='utf8')
    inputFile = open(input_file_name, 'r', encoding='utf8')
    while True:
        line = inputFile.readline() # doc_id, d_len
        fo.write(line)
        if line == '': 
            break
        line = line.strip().split()
        doc_id, d_len = int(line[0]), int(line[1])
        line = inputFile.readline() 
        # fo.write(line) # true pairs
        fo.write(str(pair_id_all_dict[doc_id])+'\n')
        if doc_id in pair_id_filtered_dict:
            fo.write(str(pair_id_filtered_dict[doc_id])+'\n') # pred pairs
        else:
            fo.write('\n')
        for i in range(d_len):
            line = inputFile.readline()
            fo.write(line)
    print ('write {} done'.format(output_file_name))


def run():
    if 'emocate' in FLAGS.scope:
        FLAGS.choose_emocate = 'use'
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    save_dir = '{}/{}/'.format(FLAGS.log_path, FLAGS.scope)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dev_eval_list, test_eval_list = [], []
    max_epoch_list = []
    cur_run = 1
    while True:
        if cur_run == FLAGS.end_run: 
            break

        print_time()
        print('\n############# run {} begin ###############'.format(cur_run))
        tf.compat.v1.reset_default_graph()

        word_idx_rev, word_idx, _, _, word_embedding, pos_embedding = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, FLAGS.path+'all_data_pair.txt', FLAGS.w2v_file)
        video_idx, video_embedding, audio_embedding = load_embedding_from_npy(FLAGS.video_idx_file, FLAGS.video_emb_file, FLAGS.audio_emb_file)

        train_file_name = 'run{}_train.txt'.format(cur_run)
        dev_file_name = 'run{}_dev.txt'.format(cur_run)
        test_file_name = 'run{}_test.txt'.format(cur_run)
        if os.path.exists(save_dir+'run1_train.txt'):
            save_dir1 = save_dir
        else:
            save_dir1 = save_dir + FLAGS.step1_file_dir # 'step1/'
        train_data = Dataset(save_dir1+train_file_name, word_idx, video_idx) 
        dev_data = Dataset(save_dir1+dev_file_name, word_idx, video_idx)
        test_data = Dataset(save_dir1+test_file_name, word_idx, video_idx)
        print('train docs: {}  dev docs: {}  test docs: {}'.format(len(train_data.x), len(dev_data.x), len(test_data.x)))

        word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
        pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')
        video_embedding = tf.constant(video_embedding, dtype=tf.float32, name='video_embedding') 
        audio_embedding = tf.constant(audio_embedding, dtype=tf.float32, name='audio_embedding') 
        embeddings = [word_embedding, pos_embedding, video_embedding, audio_embedding]

        print('\nbuild model...')
        x = tf.compat.v1.placeholder(tf.int32, [None, 2, FLAGS.max_sen_len])
        sen_len = tf.compat.v1.placeholder(tf.int32, [None, 2])
        distance = tf.compat.v1.placeholder(tf.int32, [None])
        x_emocate = tf.compat.v1.placeholder(tf.int32, [None])
        x_v = tf.compat.v1.placeholder(tf.int32, [None, 2])
        y = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.n_class])
        is_training = tf.compat.v1.placeholder(tf.float32) 

        placeholders = [x, sen_len, distance, x_emocate, x_v, y, is_training]
        
        pred_pair, reg = build_model(embeddings, placeholders)
        
        loss_op = - tf.reduce_mean(y * tf.math.log(pred_pair)) + reg * FLAGS.l2_reg
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)
        
        true_y_op = tf.argmax(y, 1)
        pred_y_op = tf.argmax(pred_pair, 1)

        acc_op = tf.reduce_mean(tf.cast(tf.equal(true_y_op, pred_y_op), tf.float32))
        print('build model done!\n')
        
        # Training Code Block
        print_info()
        tf_config = tf.compat.v1.ConfigProto()  
        tf_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=tf_config) as sess:
            while 1:
                sess.run(tf.compat.v1.global_variables_initializer())

                max_f1 = -1.
                max_epoch_index = 0
                for i in range(FLAGS.training_iter):
                    start_time, step = time.time(), 1
                    
                    # train
                    tr_predy_tofile0, tr_batch_index_list = [], []
                    for train, batch_index in get_batch_data(train_data, is_training = 1, batch_size=FLAGS.batch_size):
                        _, loss, pred_y, true_y, acc = sess.run(
                            [optimizer, loss_op, pred_y_op, true_y_op, acc_op], feed_dict=dict(zip(placeholders, train)))
                        tr_predy_tofile0.extend(list(pred_y))
                        tr_batch_index_list.extend(batch_index)
                        print('step {}: train loss {:.4f} acc {:.4f}'.format(step, loss, acc))
                        step = step + 1

                    def evaluate(test_data):
                        test = test_data.all + [0]
                        loss, te_pred_y, te_true_y = sess.run([ loss_op, pred_y_op, true_y_op], feed_dict=dict(zip(placeholders, test)))
                        if FLAGS.choose_emocate:
                            return loss, prf_2nd_step_emocate(test_data.pair_id_all, test_data.pair_id, te_pred_y), te_pred_y
                            # [f1_emo1, ..., f1_emo6, f1_avg, o_f1_emo1, ..., o_f1_emo6, o_f1_avg, keep_rate]
                        return loss, prf_2nd_step(test_data.pair_id_all, test_data.pair_id, te_pred_y), te_pred_y
                        # [p, r, f1, o_p, o_r, o_f1, keep_rate]

                    dev_loss, dev_eval, de_pred_y = evaluate(dev_data)
                    test_loss, test_eval, te_pred_y = evaluate(test_data)

                    dev_eval, test_eval = map(lambda x: np.array(x), [dev_eval, test_eval])
                    print('\nepoch {}: cost time {:.1f} s  dev_loss {:.4f}  test_loss {:.4f}\n'.format(i, time.time()-start_time, dev_loss, test_loss))
                    if FLAGS.choose_emocate:
                        # results = list(f[1:]) + [w_avg_p, w_avg_r, w_avg_f, w_avg_p_part, w_avg_r_part, w_avg_f_part] 
                        if FLAGS.emocate_eval == 4:
                            if dev_eval[11] > max_f1:
                                max_f1 = dev_eval[11]
                                max_epoch_index = i+1
                                max_dev_eval = dev_eval
                                max_test_eval = test_eval
                                tr_predy_tofile = tr_predy_tofile0
                                de_predy_tofile = de_pred_y
                                te_predy_tofile = te_pred_y
                        else:
                            if dev_eval[8] > max_f1:
                                max_f1 = dev_eval[8]
                                max_epoch_index = i+1
                                max_dev_eval = dev_eval
                                max_test_eval = test_eval
                                tr_predy_tofile = tr_predy_tofile0
                                de_predy_tofile = de_pred_y
                                te_predy_tofile = te_pred_y
                        print('dev_eval \n{}\n{} \nmax_dev_eval \n{}\n{}'.format(dev_eval[:6], dev_eval[6:], max_dev_eval[:6], max_dev_eval[6:]))
                        print('test_eval \n{}\n{} \nmax_test_eval \n{}\n{}\n\n'.format(test_eval[:6], test_eval[6:], max_test_eval[:6], max_test_eval[6:]))
                    else:
                        if dev_eval[2] > max_f1:
                            max_f1 = dev_eval[2]
                            max_epoch_index = i+1
                            max_dev_eval = dev_eval
                            max_test_eval = test_eval
                            tr_predy_tofile = tr_predy_tofile0
                            de_predy_tofile = de_pred_y
                            te_predy_tofile = te_pred_y
                        print('dev_eval: {}\nmax_dev_eval: {}\n'.format(dev_eval, max_dev_eval))
                        print('test_eval: {}\nmax_test_eval: {}\n\n'.format(test_eval, max_test_eval))
                print ('Optimization Finished!\n')
                print('############# run {} end ###############\n'.format(cur_run))
                
                if max_f1 > 0.0: # 防止有时训练不到位，F1始终为0
                    # print('train pair_id: {}  pred_y: {}'.format(len(train_data.pair_id), len(tr_predy_tofile)))
                    # print('dev pair_id: {}  pred_y: {}'.format(len(dev_data.pair_id), len(de_predy_tofile)))
                    # print('test pair_id: {}  pred_y: {}'.format(len(test_data.pair_id), len(te_predy_tofile)))
                    if FLAGS.save_pair:
                        save_pair_path = save_dir + FLAGS.log_file_name.replace('.log', '_pair/')
                        if not os.path.exists(save_pair_path):
                            os.makedirs(save_pair_path)
                        write_data_all(save_dir1+train_file_name, save_pair_path+train_file_name, train_data, tr_predy_tofile, tr_batch_index_list, FLAGS.choose_emocate)
                        write_data_all(save_dir1+dev_file_name, save_pair_path+dev_file_name, dev_data, de_predy_tofile, [], FLAGS.choose_emocate)
                        write_data_all(save_dir1+test_file_name, save_pair_path+test_file_name, test_data, te_predy_tofile, [], FLAGS.choose_emocate)
                        break


            dev_eval_list.append(max_dev_eval)
            test_eval_list.append(max_test_eval)
            max_epoch_list.append(max_epoch_index)

            print('\n--------------- previous {} runs Avg -----------------\n'.format(cur_run))
            dev_eval_list_, test_eval_list_ = map(lambda x: np.around(np.array(x), decimals=4), [dev_eval_list, test_eval_list])
            print('\ndev_eval_list: \n{}\nAvg: {}\nStd: {}\n\n'.format(dev_eval_list_, list_round(dev_eval_list_.mean(axis=0)), dev_eval_list_.std(axis=0)))
            print('\ntest_eval_list: \n{}\nAvg: {}\nStd: {}\n\n'.format(test_eval_list_, list_round(test_eval_list_.mean(axis=0)), test_eval_list_.std(axis=0)))
            print('max_epoch:\n{}  {}  {}\n\n'.format(max_epoch_list, max(max_epoch_list), np.mean(max_epoch_list)))
            print('-----------------------------------------------------')
            
            cur_run = cur_run + 1
            
    print_time()
        
     
def main(_):
    run()



if __name__ == '__main__':
    tf.compat.v1.app.run() 
