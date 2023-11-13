# encoding: utf-8

import numpy as np
import tensorflow as tf
print('\ntensorflow: {}\ntf.test.is_gpu_available: {}\n'.format(tf.__version__, tf.test.is_gpu_available()))

import sys, os, time, codecs, pdb
os.environ['KMP_WARNINGS'] = '0'

sys.path.append('./utils')
sys.path.append('./bert')
from tf_funcs import *
from pre_data_bert import *
import modeling, optimization, tokenization



FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', './data/ECF_glove_300.txt', 'embedding file')
tf.app.flags.DEFINE_string('path', './data/', 'path for dataset')
tf.app.flags.DEFINE_string('video_emb_file', './data/video_embedding_4096.npy', 'ndarray (13620, 4096)')
tf.app.flags.DEFINE_string('audio_emb_file', './data/audio_embedding_6373.npy', 'ndarray (13620, 6373)')
tf.app.flags.DEFINE_string('video_idx_file', './data/video_id_mapping.npy', 'mapping dict: {dia1utt1: 1, ...}')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 35, 'max number of tokens per sentence') 
tf.app.flags.DEFINE_integer('max_doc_len', 35, 'max number of sentences per document') 
tf.app.flags.DEFINE_integer('max_sen_len_bert', 40, 'max number of tokens per sentence') 
tf.app.flags.DEFINE_integer('max_doc_len_bert', 400, 'max number of tokens per document for Bert Model') 
## model struct ##
tf.app.flags.DEFINE_string('model_type', 'BiLSTM', 'model type: BERTcased, BERTuncased, BiLSTM')
tf.app.flags.DEFINE_string('bert_encoder_type', 'BERT_sen', 'model encoder type: BERT_doc, BERT_sen')
tf.app.flags.DEFINE_string('bert_base_dir', './data/cased_L-12_H-768_A-12/', 'base dir of pretrained bert') 
tf.app.flags.DEFINE_string('share_word_encoder', 'yes', 'whether emotion and cause share the same underlying word encoder')
tf.app.flags.DEFINE_string('choose_emocate', '', 'whether predict the emotion category')
tf.app.flags.DEFINE_string('use_x_v', '', 'whether use video embedding')
tf.app.flags.DEFINE_string('use_x_a', '', 'whether use audio embedding')
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_string('real_time', '', 'real_time conversation')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('batch_size', 8, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-5, 'learning rate')
tf.app.flags.DEFINE_integer('bert_start_idx', 20, 'bert para')
tf.app.flags.DEFINE_integer('bert_end_idx', 219, 'bert para')
tf.app.flags.DEFINE_float('bert_hidden_kb', 0.9, 'keep prob for bert')
tf.app.flags.DEFINE_float('bert_attention_kb', 0.7, 'keep prob for bert')
tf.app.flags.DEFINE_integer('end_run', 21, 'end_run')
tf.app.flags.DEFINE_float('keep_prob1', 1.0, 'keep prob for word embedding')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'keep prob for softmax layer')
tf.app.flags.DEFINE_float('keep_prob_v', 0.5, 'training dropout keep prob for visual features')
tf.app.flags.DEFINE_float('keep_prob_a', 0.5, 'training dropout keep prob for audio features')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')
tf.app.flags.DEFINE_float('emo', 1., 'loss weight of emotion ext.')
tf.app.flags.DEFINE_float('cause', 1., 'loss weight of cause ext.')
tf.app.flags.DEFINE_integer('training_iter', 15, 'number of training iter')

tf.app.flags.DEFINE_string('log_path', './log', '')
tf.app.flags.DEFINE_string('scope', 'TEMP', 'scope')
tf.app.flags.DEFINE_string('log_file_name', 'step1.log', 'name of log file')



def pre_set():
    if FLAGS.model_type=='BERTuncased':
        FLAGS.bert_base_dir = 'data/uncased_L-12_H-768_A-12/'

    if FLAGS.model_type=='BiLSTM':
        FLAGS.batch_size = 32
        FLAGS.learning_rate = 0.005
        FLAGS.keep_prob1 = 0.5
        FLAGS.training_iter = 30

def print_info():
    print('\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    print('model_type: {}\nshare_word_encoder: {}\nbert_encoder_type: {}\nchoose_emocate: {}\nvideo_emb_file: {}\naudio_emb_file: {}\nuse_x_v: {}\nuse_x_a: {}\nmax_doc_len_bert: {}\nmax_sen_len_bert {}\nreal_time: {}\n\n'.format(
        FLAGS.model_type, FLAGS.share_word_encoder, FLAGS.bert_encoder_type, FLAGS.choose_emocate, FLAGS.video_emb_file, FLAGS.audio_emb_file, FLAGS.use_x_v,  FLAGS.use_x_a, FLAGS.max_doc_len_bert, FLAGS.max_sen_len_bert, FLAGS.real_time))

    print('>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print('path: {}\nbatch: {}\nlr: {}\nkb1: {}\nkb2: {}\nl2_reg: {}\nkeep_prob_v: {}\nkeep_prob_a {}\nbert_base_dir: {}\nbert_hidden_kb: {}\nbert_attention_kb: {}\nemo: {}\ncause: {}\ntraining_iter: {}\nend_run: {}\n\n'.format(FLAGS.path, FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg, FLAGS.keep_prob_v, FLAGS.keep_prob_a, FLAGS.bert_base_dir, FLAGS.bert_hidden_kb, FLAGS.bert_attention_kb, FLAGS.emo,  FLAGS.cause, FLAGS.training_iter, FLAGS.end_run))

def build_subtasks(embeddings, placeholders):
    word_embedding, video_embedding, audio_embedding = embeddings 
    x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause, is_training = placeholders

    sen_len = tf.reshape(sen_len, [-1])
    x = tf.nn.embedding_lookup(word_embedding, x)
    x = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    x = tf.nn.dropout(x, keep_prob = is_training * FLAGS.keep_prob1 + (1.-is_training))
    
    x_a = tf.nn.embedding_lookup(audio_embedding, x_v)
    x_v = tf.nn.embedding_lookup(video_embedding, x_v)
    x_a = tf.nn.dropout(x_a, keep_prob = is_training * FLAGS.keep_prob_a + (1.-is_training))
    x_v = tf.nn.dropout(x_v, keep_prob = is_training * FLAGS.keep_prob_v + (1.-is_training))

    h2 = 2 * FLAGS.n_hidden

    def get_bert_s_ind(x_bert_sen, x_mask_bert_sen, feature_mask, is_training, scope='bert'):
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_base_dir + 'bert_config.json')
        bert_config.hidden_dropout_prob, bert_config.attention_probs_dropout_prob = 1-FLAGS.bert_hidden_kb, 1-FLAGS.bert_attention_kb
        
        x_bert_sen = tf.reshape(x_bert_sen, [-1, FLAGS.max_sen_len_bert])
        x_mask_bert_sen = tf.reshape(x_mask_bert_sen, [-1, FLAGS.max_sen_len_bert])
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=x_bert_sen,
            input_mask=x_mask_bert_sen,
            scope=scope)
        s_bert = model.get_pooled_output()

        s_bert = tf.reshape(s_bert, [-1, FLAGS.max_doc_len, s_bert.shape[-1].value])
        s_bert = s_bert * feature_mask # independent utterance representation from bert 
        return s_bert

    def get_bert_s(x_bert, x_mask_bert, x_type_bert, s_idx_bert, feature_mask, is_training, scope='bert'):
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_base_dir + 'bert_config.json')
        bert_config.hidden_dropout_prob, bert_config.attention_probs_dropout_prob = 1-FLAGS.bert_hidden_kb, 1-FLAGS.bert_attention_kb
        
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=x_bert,
            input_mask=x_mask_bert,
            token_type_ids=x_type_bert,
            scope=scope)
        s_bert = model.sequence_output
        batch_size, n_hidden = tf.shape(s_bert)[0], s_bert.shape[-1].value

        index = tf.reshape(tf.range(0, batch_size) * FLAGS.max_doc_len_bert, [batch_size, 1]) + s_idx_bert # [batch_size, FLAGS.max_doc_len]
        index = tf.reshape(index, [-1]) # [batch_size * FLAGS.max_doc_len]
        s_bert = tf.gather(tf.reshape(s_bert, [-1, n_hidden]), index)  # 取每个utt第一个[CLS]的向量
        s_bert = tf.reshape(s_bert, [-1, FLAGS.max_doc_len, n_hidden]) # [-1, FLAGS.max_doc_len, n_hidden]
        return s_bert * feature_mask

    
    def get_s(inputs, sen_len, name):
        with tf.name_scope('word_encode'): 
            inputs = biLSTM(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'_word_layer' + name) # inputs shape:  [-1, FLAGS.max_sen_len, 2 * FLAGS.n_hidden]
        with tf.name_scope('word_attention'):
            w1 = get_weight_varible('word_att_w1' + name, [h2, h2])
            b1 = get_weight_varible('word_att_b1' + name, [h2])
            w2 = get_weight_varible('word_att_w2' + name, [h2, 1])
            s = att_var(inputs,sen_len,w1,b1,w2)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, h2])
        return s
    
    def emo_cause_prediction(s_ec, is_training, name):
        s1 = tf.nn.dropout(s_ec, keep_prob = is_training * FLAGS.keep_prob2 + (1.-is_training))
        n_hidden = s1.shape[-1].value
        # print('num_hidden before prediction: ', n_hidden)
        s1 = tf.reshape(s1, [-1, n_hidden])
        pred_num = FLAGS.n_class
        if FLAGS.choose_emocate and 'emotion' in name:
            pred_num = 7
        w_ec = get_weight_varible('softmax_w_'+name, [n_hidden, pred_num])
        b_ec = get_weight_varible('softmax_b_'+name, [pred_num])
        pred_ec = tf.nn.softmax(tf.matmul(s1, w_ec) + b_ec)
        pred_ec = tf.reshape(pred_ec, [-1, FLAGS.max_doc_len, pred_num])
        return pred_ec, w_ec, b_ec
    
    def concate_feature(s1, x_v, x_a):
        if FLAGS.use_x_v:
            s1 = tf.concat([s1, x_v], axis = 2)
        if FLAGS.use_x_a:
            s1 = tf.concat([s1, x_a], axis = 2)
        return s1

    cause_list, emo_list, reg = [], [], 0
    feature_mask = getmask(doc_len, FLAGS.max_doc_len, [-1, FLAGS.max_doc_len, 1])

    x_v_c = x_v
    x_v = tf.nn.relu(layer_normalize(tf.layers.dense(x_v, h2, use_bias=True)))
    x_v_c = tf.nn.relu(layer_normalize(tf.layers.dense(x_v_c, h2, use_bias=True)))
    pred_emo_x_v, w_emo_x_v, b_emo_x_v = emo_cause_prediction(x_v, is_training, name='emotion_x_v')

    x_a_c = x_a
    x_a = tf.nn.relu(layer_normalize(tf.layers.dense(x_a, h2, use_bias=True)))
    x_a_c = tf.nn.relu(layer_normalize(tf.layers.dense(x_a_c, h2, use_bias=True)))
    pred_emo_x_a, w_emo_x_a, b_emo_x_a = emo_cause_prediction(x_a, is_training, name='emotion_x_a')

    if FLAGS.model_type=='BiLSTM':
        s1_emo = get_s(x, sen_len, name='_word_encode_emo')
        if FLAGS.share_word_encoder:
            s1_cause = s1_emo
        else:
            s1_cause = get_s(x, sen_len, name='_word_encode_cause')
        s1_emo = concate_feature(s1_emo, x_v, x_a)
        s1_cause = concate_feature(s1_cause, x_v_c, x_a_c)
        
        if FLAGS.real_time:
            s_emo = LSTM(s1_emo, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + '_sentence_encode_emo')
            s_cause = LSTM(s1_cause, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + '_sentence_encode_cause')
        else:
            s_emo = biLSTM(s1_emo, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + '_sentence_encode_emo')
            s_cause = biLSTM(s1_cause, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + '_sentence_encode_cause')

    else:
        if FLAGS.bert_encoder_type == 'BERT_doc':
            s_bert_emo = get_bert_s(x_bert, x_mask_bert, x_type_bert, s_idx_bert, feature_mask, is_training, scope = 'bert_emotion')
            if FLAGS.share_word_encoder:
                s_bert_cause = s_bert_emo
            else:
                s_bert_cause = get_bert_s(x_bert, x_mask_bert, x_type_bert, s_idx_bert, feature_mask, is_training, scope = 'bert_cause')
        else:
            s_bert_emo = get_bert_s_ind(x_bert_sen, x_mask_bert_sen, feature_mask, is_training, scope = 'bert_emotion')
            if FLAGS.share_word_encoder:
                s_bert_cause = s_bert_emo
            else:
                s_bert_cause = get_bert_s_ind(x_bert_sen, x_mask_bert_sen, feature_mask, is_training, scope = 'bert_cause')

        s_emo = concate_feature(s_bert_emo, x_v, x_a)
        s_emo = s_emo * feature_mask 
        s_emo = tf.layers.dense(s_emo, h2, use_bias=True) 
        
        s_cause = concate_feature(s_bert_cause, x_v_c, x_a_c) 
        s_cause = s_cause * feature_mask
        s_cause = tf.layers.dense(s_cause, h2, use_bias=True)

        if FLAGS.real_time:
            trans = standard_trans_realtime
        else:
            trans = standard_trans

        s_emo = trans(s_emo, h2, n_head=1, scope=FLAGS.scope + 'sentence_encode_emo_')
        s_cause = trans(s_cause, h2, n_head=1, scope=FLAGS.scope + 'sentence_encode_cause_')

    pred_emo, w_emo, b_emo = emo_cause_prediction(s_emo, is_training, name='emotion')
    pred_cause, w_cause, b_cause = emo_cause_prediction(s_cause, is_training, name='cause')

    reg = tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)
    reg += tf.nn.l2_loss(w_emo) + tf.nn.l2_loss(b_emo)
    # reg += tf.nn.l2_loss(w_emo_x_v) + tf.nn.l2_loss(b_emo_x_v)
    # reg += tf.nn.l2_loss(w_emo_x_a) + tf.nn.l2_loss(b_emo_x_a)
    return pred_emo, pred_emo_x_v, pred_emo_x_a, pred_cause, s_emo, s_cause, reg



def build_model(embeddings, placeholders):
    ########################################## emotion & cause extraction  ############
    print('building subtasks')
    pred_emo, pred_emo_x_v, pred_emo_x_a, pred_cause, s_emo, s_cause, reg = build_subtasks(embeddings, placeholders)
    print('build subtasks Done!')

    return pred_emo, pred_emo_x_v, pred_emo_x_a, pred_cause, reg


class Dataset(object):
    def __init__(self, data_file_name, tokenizer, word_idx, video_idx, spe_idx):
        x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause, doc_id, y_pairs = load_data_utt_step1(data_file_name, tokenizer, word_idx, video_idx, spe_idx, FLAGS.max_doc_len, FLAGS.max_sen_len, FLAGS.max_doc_len_bert, FLAGS.max_sen_len_bert, FLAGS.model_type, FLAGS.choose_emocate)

        self.x_bert_sen, self.x_mask_bert_sen = x_bert_sen, x_mask_bert_sen
        self.x_bert, self.x_mask_bert, self.x_type_bert, self.s_idx_bert = x_bert, x_mask_bert, x_type_bert, s_idx_bert
        self.x, self.sen_len, self.doc_len, self.speaker, self.x_v = x, sen_len, doc_len, speaker, x_v
        self.y_emotion, self.y_cause  = y_emotion, y_cause
        self.all = [x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause]
        self.doc_id, self.y_pairs = doc_id, y_pairs
        

def get_batch_data(dataset, is_training, batch_size):
    test = bool(1 - is_training)
    for index in batch_index(len(dataset.y_cause), batch_size, test):
        feed_list = list(map(lambda x: x[index], dataset.all)) + [is_training]
        yield feed_list, len(index)

def run():
    pre_set()
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    save_dir = '{}/{}/'.format(FLAGS.log_path, FLAGS.scope)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    emo_list, cause_list, emo_emocate_list = [], [], []
    emo_max_epoch_list, cau_max_epoch_list = [],[]
    cur_run = 1
    while True:
        if cur_run == FLAGS.end_run: 
            break

        print_time()
        print('\n############# run {} begin ###############'.format(cur_run))
        tf.compat.v1.reset_default_graph()
        
        word_idx_rev, word_idx, spe_idx_rev, spe_idx, word_embedding, _ = load_w2v(FLAGS.embedding_dim, FLAGS.embedding_dim_pos, FLAGS.path+'all_data_pair.txt', FLAGS.w2v_file)
        video_idx, video_embedding, audio_embedding = load_embedding_from_npy(FLAGS.video_idx_file, FLAGS.video_emb_file, FLAGS.audio_emb_file)
        
        def get_bert_tokenizer():
            do_lower_case = True if 'uncased' in FLAGS.bert_base_dir else False
            tokenization.validate_case_matches_checkpoint(do_lower_case, FLAGS.bert_base_dir + 'bert_model.ckpt')
            return tokenization.FullTokenizer(vocab_file = FLAGS.bert_base_dir + 'vocab.txt', do_lower_case = do_lower_case)
        tokenizer = get_bert_tokenizer()
        
        train_data = Dataset(FLAGS.path+'train.txt', tokenizer, word_idx, video_idx, spe_idx)
        dev_data = Dataset(FLAGS.path+'dev.txt', tokenizer, word_idx, video_idx, spe_idx)
        test_data = Dataset(FLAGS.path+'test.txt', tokenizer, word_idx, video_idx, spe_idx)
        print('train docs: {}  dev docs: {}  test docs: {}'.format(len(train_data.x), len(dev_data.x), len(test_data.x)))

        word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
        video_embedding = tf.constant(video_embedding, dtype=tf.float32, name='video_embedding') 
        audio_embedding = tf.constant(audio_embedding, dtype=tf.float32, name='audio_embedding') 
        embeddings = [word_embedding, video_embedding, audio_embedding]

        print('\nbuild model...')
        x = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
        doc_len = tf.compat.v1.placeholder(tf.int32, [None])
        sen_len = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len])
        x_v = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len])
        speaker = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len])
        # for label
        pred_num = 2
        if FLAGS.choose_emocate:
            pred_num = 7
        y_emotion = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.max_doc_len, pred_num])
        y_cause = tf.compat.v1.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
        is_training = tf.compat.v1.placeholder(tf.float32) 
        # for Bert_sen
        x_bert_sen = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len_bert])
        x_mask_bert_sen = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len_bert])
        # for Bert_doc
        x_bert = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len_bert])  
        x_mask_bert = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len_bert]) 
        x_type_bert = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len_bert]) 
        s_idx_bert = tf.compat.v1.placeholder(tf.int32, [None, FLAGS.max_doc_len]) 
        
        placeholders = [x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause, is_training]  
        
        pred_emo, pred_emo_x_v, pred_emo_x_a, pred_cause, reg = build_model(embeddings, placeholders)
        print('build model done!\n')

        loss_emo = - tf.reduce_sum(y_emotion * tf.math.log(pred_emo)) / tf.cast(tf.reduce_sum(y_emotion), dtype=tf.float32)
        loss_cause = - tf.reduce_sum(y_cause * tf.math.log(pred_cause)) / tf.cast(tf.reduce_sum(y_cause), dtype=tf.float32)
        loss_op = loss_cause * FLAGS.cause + loss_emo * FLAGS.emo + reg * FLAGS.l2_reg

        if FLAGS.use_x_a:
            loss_emo_x_a = - tf.reduce_sum(y_emotion * tf.math.log(pred_emo_x_a)) / tf.cast(tf.reduce_sum(y_emotion), dtype=tf.float32)
            loss_op += loss_emo_x_a
        
        def get_bert_optimizer(loss_op):
            num_train_steps = int(len(train_data.x) / FLAGS.batch_size * FLAGS.training_iter)
            num_warmup_steps = int(num_train_steps * 0.1)
            optimizer, run_lr = optimization.create_optimizer_dzx(loss_op, FLAGS.learning_rate, num_train_steps, num_warmup_steps, FLAGS.bert_start_idx, FLAGS.bert_end_idx, False)
            return optimizer, run_lr

        def init_from_bert_checkpoint():
            init_checkpoint = FLAGS.bert_base_dir + 'bert_model.ckpt'
            tvars = tf.compat.v1.trainable_variables()
            (assignment_map_emotion, initialized_variable_names_emotion) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, bert_scope='bert_emotion')
            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map_emotion) 

            tf.compat.v1.logging.info("**** Trainable Variables ****")
            if not FLAGS.share_word_encoder:
                (assignment_map_cause, initialized_variable_names_cause) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, bert_scope='bert_cause')
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map_cause)
                initialized_variable_names_emotion.update(initialized_variable_names_cause)

            # 仅仅只是替换变量的 initializers，后面运行sess.run(tf.global_variables_initializer())时才真正的加载
            initialized_variable_names = initialized_variable_names_emotion 
            for idx, var in enumerate(tvars):
                init_string = ", *INIT_FROM_CKPT*" if var.name in initialized_variable_names else ""
                print("var-index {}:  name = {}, shape = {}{}".format(idx, var.name, var.shape, init_string))

        if FLAGS.model_type=='BiLSTM':
            optimizer, run_lr = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op), tf.constant(FLAGS.learning_rate)
        else:
            optimizer, run_lr = get_bert_optimizer(loss_op)
        
        true_y_emo_op = tf.argmax(y_emotion, 2)
        pred_y_emo_op = tf.argmax(pred_emo, 2)
        pred_emo_x_v_op = tf.argmax(pred_emo_x_v, 2)
        pred_emo_x_a_op = tf.argmax(pred_emo_x_a, 2)
        true_y_cause_op = tf.argmax(y_cause, 2)
        pred_y_cause_op = tf.argmax(pred_cause, 2)


        # Training Code Block
        print_info()
        tf_config = tf.compat.v1.ConfigProto()  
        tf_config.gpu_options.allow_growth = True # 程序按需申请内存
        with tf.compat.v1.Session(config=tf_config) as sess:
            init_from_bert_checkpoint()
            sess.run(tf.compat.v1.global_variables_initializer())

            max_f1_emo, max_f1_cause = [-1.] * 2
            max_f1_emo_emocate = [-1.]
            max_epoch_index_emo, max_epoch_index_cause  = 0, 0
            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                test_results = []
                
                def combine_result(x):
                    ret = []
                    for a in list(x):
                        ret.extend(list(a))
                    return np.array(ret) 
                
                # train                        
                for train, tr_index in get_batch_data(train_data, is_training = 1, batch_size=FLAGS.batch_size):
                    _, run_lr_tmp, loss, loss_e, loss_c, pred_y_cause, true_y_cause, pred_y_emo, pred_emo_x_v, pred_emo_x_a, true_y_emo, doc_len_batch = sess.run(
                        [optimizer, run_lr, loss_op, loss_emo, loss_cause, pred_y_cause_op, true_y_cause_op, pred_y_emo_op, pred_emo_x_v_op, pred_emo_x_a_op, true_y_emo_op, doc_len], feed_dict=dict(zip(placeholders, train))) 

                    loss_all = list_round([loss, loss_e, loss_c])
                    if step % 10 == 0:
                        print('step {}: train loss {:.4f} {:.4f} {:.4f} run_lr {:.6f}'.format(step,loss_all[0],loss_all[1],loss_all[2], run_lr_tmp))
                        p, r, f1 = cal_prf(pred_y_emo, true_y_emo, doc_len_batch)
                        print('emotion_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                        p, r, f1 = cal_prf(pred_y_cause, true_y_cause, doc_len_batch)
                        print('cause_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                        
                        if FLAGS.choose_emocate:
                            emo_f1_emocate = cal_prf_emocate(pred_y_emo, true_y_emo, doc_len_batch)
                            print('emotion_prediction_emocate: train f1 {} '.format(np.around(emo_f1_emocate, decimals=2)))
                        print('\ncost time: {:.1f}s\n'.format(time.time()-start_time))
                        start_time = time.time()
                    test_results.append([pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, doc_len_batch])
                    if step % 60 == 0:
                        test_results = zip(*test_results)
                        pred_y_cause, true_y_cause, pred_y_emo, true_y_emo, doc_len_batch = map(combine_result, test_results)
                        test_results = []
                        print('\n############## Evaluation on {}-{} train steps ##############'.format(step-60,step))
                        p, r, f1 = cal_prf(pred_y_emo, true_y_emo, doc_len_batch)
                        print('emotion_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                        p, r, f1 = cal_prf(pred_y_cause, true_y_cause, doc_len_batch)
                        print('cause_prediction: train p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1 ))
                    step = step + 1

                # dev
                test_results = []
                for test, _ in get_batch_data(dev_data, is_training = 0, batch_size=FLAGS.batch_size):
                    test_results_tmp = sess.run(
                        [loss_op, pred_y_cause_op, true_y_cause_op, pred_y_emo_op, pred_emo_x_v_op, pred_emo_x_a_op, true_y_emo_op, doc_len], feed_dict=dict(zip(placeholders, test)))
                    test_results.append(test_results_tmp)
                
                test_results = list(zip(*test_results))
                de_loss = np.array(test_results[0]).mean()
                de_pred_y_cause, de_true_y_cause, de_pred_y_emo, de_pred_emo_x_v, de_pred_emo_x_a, de_true_y_emo, de_doc_len_batch = map(combine_result, test_results[1:])

                # test
                test_results = []
                for test, _ in get_batch_data(test_data, is_training = 0, batch_size=FLAGS.batch_size):
                    test_results_tmp = sess.run(
                        [loss_op, loss_emo, loss_cause, pred_y_cause_op, true_y_cause_op, pred_y_emo_op, pred_emo_x_v_op, pred_emo_x_a_op, true_y_emo_op, doc_len], feed_dict=dict(zip(placeholders, test)))
                    test_results.append(test_results_tmp)
                
                test_results = list(zip(*test_results))
                te_loss, te_loss_e, te_loss_c = np.array(test_results[0]).mean(), np.array(test_results[1]).mean(), np.array(test_results[2]).mean()
                te_pred_y_cause, te_true_y_cause, te_pred_y_emo, te_pred_emo_x_v, te_pred_emo_x_a, te_true_y_emo, te_doc_len_batch = map(combine_result, test_results[3:])
                
                te_loss_all = list_round([te_loss, te_loss_e, te_loss_c])
                print('\nepoch {}: test loss {:.4f} {:.4f} {:.4f} cost time: {:.1f}s\n'.format(i,te_loss_all[0],te_loss_all[1],te_loss_all[2], time.time()-start_time))
                
                if FLAGS.choose_emocate:
                    de_f1_emo_emocate = cal_prf_emocate(de_pred_y_emo, de_true_y_emo, de_doc_len_batch)
                    te_f1_emo_emocate = cal_prf_emocate(te_pred_y_emo, te_true_y_emo, te_doc_len_batch)
                    
                    if de_f1_emo_emocate[-1] > max_f1_emo_emocate[-1]:
                        max_f1_emo_emocate = de_f1_emo_emocate
                        te_max_f1_emo_emocate = te_f1_emo_emocate
                        max_epoch_index_emo = i+1
                        de_predemo_tofile = de_pred_y_emo
                        te_predemo_tofile = te_pred_y_emo
                        tr_predemo_tofile = []
                        for train, _ in get_batch_data(train_data, is_training = 0, batch_size=FLAGS.batch_size):
                            pred_y_emo = sess.run(pred_y_emo_op, feed_dict=dict(zip(placeholders, train)))
                            tr_predemo_tofile.extend(list(pred_y_emo))
                        
                    print('emotion_prediction_emocate: \ndev f1 {}'.format(np.around(de_f1_emo_emocate, decimals=4)))
                    print('dev max_f1 {}'.format(np.around(max_f1_emo_emocate, decimals=4)))
                    print('test f1 {}'.format(np.around(te_f1_emo_emocate, decimals=4)))
                    print('test max_f1 {}\n'.format(np.around(te_max_f1_emo_emocate, decimals=4)))
                else:
                    de_p, de_r, de_f1 = cal_prf(de_pred_y_emo, de_true_y_emo, de_doc_len_batch)
                    te_p, te_r, te_f1 = cal_prf(te_pred_y_emo, te_true_y_emo, te_doc_len_batch)
                    de_prf_emo = [de_p, de_r, de_f1]
                    te_prf_emo = [te_p, te_r, te_f1]
                    
                    if de_f1 > max_f1_emo:
                        max_p_emo, max_r_emo, max_f1_emo = de_p, de_r, de_f1
                        te_max_p_emo, te_max_r_emo, te_max_f1_emo = te_p, te_r, te_f1
                        max_epoch_index_emo = i+1
                        de_predemo_tofile = de_pred_y_emo
                        te_predemo_tofile = te_pred_y_emo
                        tr_predemo_tofile = []
                        for train, _ in get_batch_data(train_data, is_training = 0, batch_size=FLAGS.batch_size):
                            pred_y_emo = sess.run(pred_y_emo_op, feed_dict=dict(zip(placeholders, train)))
                            tr_predemo_tofile.extend(list(pred_y_emo))

                    print('emotion_prediction: \ndev p {:.4f} r {:.4f} f1 {:.4f}'.format(de_p, de_r, de_f1 ))
                    print('dev max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(max_p_emo, max_r_emo, max_f1_emo))
                    print('test p {:.4f} r {:.4f} f1 {:.4f}'.format(te_p, te_r, te_f1 ))
                    print('test max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(te_max_p_emo, te_max_r_emo, te_max_f1_emo))

                de_p, de_r, de_f1 = cal_prf(de_pred_y_cause, de_true_y_cause, de_doc_len_batch)
                de_prf_cause = [de_p, de_r, de_f1]
                te_p, te_r, te_f1 = cal_prf(te_pred_y_cause, te_true_y_cause, te_doc_len_batch)
                te_prf_cause = [te_p, te_r, te_f1]
                    
                if de_f1 > max_f1_cause:
                    max_epoch_index_cause = i+1
                    max_p_cause, max_r_cause, max_f1_cause = de_p, de_r, de_f1
                    te_max_p_cause, te_max_r_cause, te_max_f1_cause = te_p, te_r, te_f1

                    de_predcause_tofile = de_pred_y_cause
                    te_predcause_tofile = te_pred_y_cause
                    tr_predcause_tofile = []
                    for train, _ in get_batch_data(train_data, is_training = 0, batch_size=FLAGS.batch_size):
                        pred_y_cause = sess.run(pred_y_cause_op, feed_dict=dict(zip(placeholders, train)))
                        tr_predcause_tofile.extend(list(pred_y_cause))

                print('cause_prediction: \ndev p {:.4f} r {:.4f} f1 {:.4f}'.format(de_p, de_r, de_f1 ))
                print('dev max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(max_p_cause, max_r_cause, max_f1_cause))
                print('test p {:.4f} r {:.4f} f1 {:.4f}'.format(te_p, te_r, te_f1 ))
                print('test max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(te_max_p_cause, te_max_r_cause, te_max_f1_cause))

            print('Optimization Finished!\n')
            print('############# run {} end ###############\n'.format(cur_run))

            def write_data(file_name, dataset, pred_y_emo, pred_y_cause):
                doc_id, y_pairs, x, sen_len, doc_len, speaker, y_emotion = dataset.doc_id, dataset.y_pairs, dataset.x, dataset.sen_len, dataset.doc_len, dataset.speaker, dataset.y_emotion
                y_emotion = np.argmax(y_emotion, 2)
                emotion_idx_rev = dict(zip(range(7), ['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']))
                g = open(file_name, 'w', encoding='utf8')
                for i in range(len(doc_id)):
                    g.write(doc_id[i]+' '+str(doc_len[i])+'\n')
                    g.write(str(y_pairs[i])+'\n')
                    for j in range(doc_len[i]):
                        utterance = ''
                        for k in range(sen_len[i][j]):
                            utterance = utterance + word_idx_rev[x[i][j][k]] + ' '
                        g.write('{} | {} | {} | {} | {} | {}\n'.format(j+1, pred_y_emo[i][j], pred_y_cause[i][j], spe_idx_rev[speaker[i][j]], emotion_idx_rev[y_emotion[i][j]], utterance))
                print('write {} done'.format(file_name))
            train_file_name = 'run{}_train.txt'.format(cur_run)
            dev_file_name = 'run{}_dev.txt'.format(cur_run)
            test_file_name = 'run{}_test.txt'.format(cur_run)
            save_dir1 = save_dir + FLAGS.log_file_name.replace('.log', '/')
            if not os.path.exists(save_dir1):
                os.makedirs(save_dir1)
            write_data(save_dir1 + train_file_name, train_data, tr_predemo_tofile, tr_predcause_tofile)
            write_data(save_dir1 + dev_file_name, dev_data, de_predemo_tofile, de_predcause_tofile)
            write_data(save_dir1 + test_file_name, test_data, te_predemo_tofile, te_predcause_tofile)

            if FLAGS.choose_emocate:
                emo_emocate_list.append(te_max_f1_emo_emocate)
            else:
                emo_list.append([te_max_p_emo, te_max_r_emo, te_max_f1_emo])
            cause_list.append([te_max_p_cause, te_max_r_cause, te_max_f1_cause])
            emo_max_epoch_list.append(max_epoch_index_emo)
            cau_max_epoch_list.append(max_epoch_index_cause)

            cur_run = cur_run + 1

            print('\n--------- Previous {} runs ---------\n'.format(cur_run-1))
            

            if FLAGS.choose_emocate:
                emo_emocate_list_1 = np.array(emo_emocate_list)
                print('\nemotion_prediction_emocate: test f1 in {} run: {}'.format(cur_run-1, np.around(emo_emocate_list_1, decimals=4)))
                avg_emo_emocate = emo_emocate_list_1.mean(axis=0)
                std_emo_emocate = emo_emocate_list_1.std(axis=0)
                print('average : f1 {}'.format(np.around(avg_emo_emocate, decimals=4)))
                print('std : f1 {}\n'.format(np.around(std_emo_emocate, decimals=4)))
            else:
                emo_list_1 = np.array(emo_list)
                print('\nemotion_prediction: test f1 in {} run: {}'.format(cur_run-1, emo_list_1[:,2:]))
                emo_p, emo_r, emo_f1 = emo_list_1.mean(axis=0)
                print('average prf: {:.4f} {:.4f} {:.4f} \nstd_f1 {:.4f} \n'.format(emo_p, emo_r, emo_f1, emo_list_1.std(axis=0)[2]))

            cause_list_1 = np.array(cause_list)
            print('\ncause_prediction: test f1 in {} run: {}'.format(cur_run-1, cause_list_1[:,2:]))
            cause_p, cause_r, cause_f1 = cause_list_1.mean(axis=0)
            print('average prf: {:.4f} {:.4f} {:.4f} \nstd_f1 {:.4f} \n'.format(cause_p, cause_r, cause_f1, cause_list_1.std(axis=0)[2]))

            print('max_epoch_emo:\n{}  {}  {}\nmax_epoch_cause:\n{}  {}  {}\n'.format(emo_max_epoch_list, max(emo_max_epoch_list), np.mean(emo_max_epoch_list), cau_max_epoch_list, max(cau_max_epoch_list), np.mean(cau_max_epoch_list)))
            print('-----------------------------------------------------')

    print_time()



def main(_):

    run()


    

if __name__ == '__main__':
    tf.compat.v1.app.run() 
