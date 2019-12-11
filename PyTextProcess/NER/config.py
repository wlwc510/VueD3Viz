# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_string("src_file", './resource/组合/da_data.txt', "Training data.")  # 组合 属于 (原始 | 换词 | 组合 | 换词+组合)
tf.app.flags.DEFINE_string("tgt_file", './resource/组合/da_data_tag.txt', "labels.")

tf.app.flags.DEFINE_string("model_path", './resource/组合/model1_2/', "model save path")  # model model1_2 model1_5
tf.app.flags.DEFINE_string("save_pred", './resource/组合/predict_tag1_2.txt', "predict save path")  #

tf.app.flags.DEFINE_string("pred_file", './resource/predict.txt', "prediction data.")
tf.app.flags.DEFINE_string("pred_file_target", './resource/predict_tag.txt', "prediction target.")

tf.app.flags.DEFINE_string("src_vocab_file", './resource/source_vocab.txt', "source vocabulary.")
tf.app.flags.DEFINE_string("tgt_vocab_file", './resource/target_vocab.txt', "targets.")

tf.app.flags.DEFINE_string("word_embedding_file", './resource/word_vec/word_vec.txt', "extra word embeddings.") # 不变

# 词典的路径
tf.app.flags.DEFINE_string("dict_rs", './resource/词典/dict_rs.txt', "remote sensing word")
tf.app.flags.DEFINE_string("dict_act", './resource/词典/dict_act.txt', "action word")
tf.app.flags.DEFINE_string("dict_apl", './resource/词典/dict_apl.txt', "application word")
tf.app.flags.DEFINE_string("dict_rs_tag", './resource/词典/dict_rs_tag.txt', "remote sensing tag")
tf.app.flags.DEFINE_string("dict_act_tag", './resource/词典/dict_act_tag.txt', "action tag")
tf.app.flags.DEFINE_string("dict_apl_tag", './resource/词典/dict_apl_tag.txt', "application tag")


# 这里默认词向量的维度是300, 如果自行训练的词向量维度不是300,则需要改这里的值。
tf.app.flags.DEFINE_integer("embeddings_size", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("max_sequence", 256, "max sequence length.")

tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.app.flags.DEFINE_integer("epoch", 5000, "epoch.")
tf.app.flags.DEFINE_float("dropout", 0.8, "drop out")

tf.app.flags.DEFINE_string("action", 'train', "train | predict")
FLAGS = tf.app.flags.FLAGS
