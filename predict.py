#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import os
from itertools import izip
import numpy as np
import tensorflow as tf

import data_utils
import tfrecords_utils
import model
import train

FLAGS = tf.app.flags.FLAGS


def predict(words_list):
  num_classes = FLAGS.num_classes
  num_layers = FLAGS.num_layers
  num_steps = FLAGS.num_steps
  embedding_size = FLAGS.embedding_size
  hidden_size = FLAGS.hidden_size
  keep_prob = FLAGS.keep_prob
  vocab_size = FLAGS.vocab_size
  vocab_path = FLAGS.vocab_path

  words_vocab = data_utils.initialize_vocabulary(os.path.join(vocab_path, 'words_vocab.txt'))
  labels_vocab = data_utils.initialize_vocabulary(os.path.join(vocab_path, 'labels_vocab.txt'))
  labels_vocab = dict(izip(labels_vocab.itervalues(), labels_vocab.iterkeys()))

  inputs_data_list = []
  for words in words_list:
    word_ids = data_utils.sentence_to_ids(words, words_vocab)
    word_ids_str = ' '.join([str(word) for word in word_ids])
    word_ids_padding = data_utils.align_word(word_ids_str, num_steps)
    inputs_data = np.array([int(tok) for tok in word_ids_padding.strip().split() if tok])
    inputs_data_list.append(inputs_data)

  predict_batch_size = len(inputs_data_list)
  inputs_placeholder = tf.placeholder(tf.int64, [None, num_steps])

  checkpoint_dir = FLAGS.checkpoint_dir
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  with tf.variable_scope('model', reuse=None):
    logits, final_state = model.inference(inputs_placeholder, predict_batch_size, num_steps, vocab_size, embedding_size,
                                          hidden_size, keep_prob, num_layers, num_classes, is_training=False)
  softmax_result = tf.nn.softmax(logits)
  predict_result = tf.arg_max(softmax_result, 1)

  with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print('read model from {}'.format(ckpt.model_checkpoint_path))
      saver = tf.train.Saver()
      saver.restore(sess, ckpt.model_checkpoint_path)
      result_list = sess.run(predict_result, feed_dict={inputs_placeholder: inputs_data_list})
    assert len(words_list) == len(result_list)
    predict_label_list = []
    for (words, result) in zip(words_list, result_list):
      print words.decode('string_escape')
      label_ids = result
      label = labels_vocab.get(int(label_ids), data_utils.UNK_ID)
      print label
      predict_label_list.append(label)
    return predict_label_list
    print 'predict result : '
    print result


def main(_):
  raw_word = ['外婆家',
              '如 家 快捷酒店',
              '我们 相爱 吧 第二季',
              '现为 中国作家协会 会员']
  result = predict(raw_word)


if __name__ == '__main__':
  tf.app.run()
