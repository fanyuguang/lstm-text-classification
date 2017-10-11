#!/usr/bin/env Python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import os
import sys
import re
import random
from collections import Counter
import jieba
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('raw_data_path', 'data/raw-data/', 'raw data directory')
tf.app.flags.DEFINE_string('segment_path', 'data/segment/', 'word segment directory')
tf.app.flags.DEFINE_string('train_path', 'data/train/', 'train directory')
tf.app.flags.DEFINE_string('ids_path', 'data/ids/', 'ids directory')
tf.app.flags.DEFINE_string('vocab_path', 'data/vocab/', 'vocab directory')
tf.app.flags.DEFINE_integer('vocab_size', 800000, 'vocab size')
tf.app.flags.DEFINE_float('train_percent', 0.8, 'train percent')
tf.app.flags.DEFINE_float('val_percent', 0.1, 'val test percent')

# _WORD_SPLIT = re.compile(r'([.,!?\"':;)(-_=+|`~@#$%^&*][}{。，！？、“‘：：;`～@#￥%……&×）（「·}{]©)')
_WORD_SPLIT = re.compile(u'[^A-Za-z0-9\u4e00-\u9fa5]')
_SENTENCE_SPLIT = re.compile(u'[.,!?。，！？\t]')

# Special vocabulary symbols - we always put them at the start.
_PAD = b'_PAD'
_GO = b'_GO'
_EOS = b'_EOS'
_UNK = b'_UNK'
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def split(sentence, split_token):
  if not isinstance(sentence, unicode):
    sentence = sentence.decode()
  word_list = split_token.split(sentence.strip())
  return [word for word in word_list if word]


def sentence_segment(sentence):
  word_list = jieba.cut(sentence.strip())
  split_word_list = []
  for word in word_list:
    split_word = split(word, split_token=_WORD_SPLIT)
    if split_word:
      split_word_list.extend(split_word)
  return split_word_list


def file_segment(data_path, data_segment_path):
  print 'Starting word segmentation ' + data_path
  for sub_data_filename in os.listdir(data_path):
    sub_data_segment_path = os.path.join(data_segment_path,
                                         (sub_data_filename[:sub_data_filename.index('.')] + '_segment.txt'))
    file_data_segment = open(sub_data_segment_path, mode='w')
    sub_data_path = os.path.join(data_path, sub_data_filename)
    with open(sub_data_path, mode='r') as file_data:
      for sentence in file_data:
        split_sentence_list = split(sentence, split_token=_SENTENCE_SPLIT)
        for split_sentence in split_sentence_list:
          word_list = sentence_segment(split_sentence)
          if word_list:
            words = ' '.join(word_list)
            file_data_segment.write(words + '\n')
            print words
    file_data_segment.close()


def count_vocabulary(data_path):
  vocab = {}
  with open(data_path, mode='r') as data_file:
    for line in data_file:
      word_list = line.split()
      for word in word_list:
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1
        print word.decode('string_escape') + ' ',
      print ''
  return vocab


def create_vocabulary(data_path, vocab_path, vocab_size):
  print 'Creating vocabulary' + vocab_path

  def counter_add(a, b):
    a.update(b)
    return a

  data_path_list = [os.path.join(data_path, sub_data_filename) for sub_data_filename in os.listdir(data_path)]
  vocab = reduce(counter_add, (Counter(count_vocabulary(sub_data_path)) for sub_data_path in data_path_list))
  vocab = dict(vocab)
  vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
  if len(vocab_list) > vocab_size:
    vocab_list = vocab_list[:vocab_size]
  with open(vocab_path, mode='w') as vocab_file:
    for word in vocab_list:
      vocab_file.write(word + '\n')
  vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
  return vocab


def initialize_vocabulary(vocab_path):
  print 'Initialize vocabulary' + vocab_path
  if os.path.exists(vocab_path):
    rev_vocab = []
    with open(vocab_path, mode='r') as vocabulary_file:
      rev_vocab.extend(vocabulary_file.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab
  else:
    raise ValueError('Vocabulary file %s not found.', vocab_path)


def sentence_to_ids(sentence, vocab):
  word_list = sentence.split()
  return [vocab.get(word, UNK_ID) for word in word_list]


def file_to_ids(data_path, ids_path, vocab):
  print 'Tokenizing data in ' + data_path
  ids_list = []
  with open(data_path, mode='r') as data_file:
    for line in data_file:
      ids = sentence_to_ids(line, vocab)
      if ids:
        ids_list.append(' '.join([str(tok) for tok in ids]))
      else:
        ids_list.append('')
  with open(ids_path, mode='w') as ids_file:
    for ids in ids_list:
      ids_file.write(ids + '\n')
  return ids_list


def align_word(words, vector_size):
  word_list = words.strip().split()
  words_count = len(word_list)
  if words_count < vector_size:
    padding = ' '.join([str(PAD_ID) for _ in range(vector_size - words_count)])
    if words_count:
      return (words + ' ' + padding)
    else:
      return padding
  else:
    words_padding = ' '.join([word for index, word in enumerate(word_list) if index < vector_size])
    return words_padding


def align_ids(ids_list, vector_size):
  print 'Align data'
  ids_padding_list = []
  for ids in ids_list:
    ids_padding = align_word(ids, vector_size)
    ids_padding.append(ids_padding)
  return ids_padding_list


def prepare_datasets(data_path, ids_path, vocab_path, words_vocab, train_percent, val_percent):
  word_ids_list = []
  label_ids_list = []
  label_ids = 0
  label_vocab_file = open(os.path.join(vocab_path, 'labels_vocab.txt'), 'w')
  for data_filename in os.listdir(data_path):
    label_name = data_filename[:data_filename.index('.')]
    sub_data_path = os.path.join(data_path, data_filename)
    sub_ids_path = os.path.join(ids_path, (label_name + '_ids.txt'))
    ids_list = file_to_ids(sub_data_path, sub_ids_path, words_vocab)
    ids_list = [ids for ids in ids_list if ids]
    word_ids_list.extend(ids_list)
    ids_list_len = len(ids_list)
    label_ids_list.extend([label_ids for index in range(ids_list_len)])
    label_vocab_file.write(label_name + '\n')
    label_ids += 1
  label_vocab_file.close()

  word_label_zip = zip(word_ids_list, label_ids_list)
  random.shuffle(word_label_zip)
  shuffle_word_ids_list, shuffle_label_ids_list = zip(*word_label_zip)

  data_size = len(word_ids_list)
  train_validation_index = int(data_size * train_percent)
  validation_test_index = int(data_size * (train_percent + val_percent))
  train_word_ids_list = shuffle_word_ids_list[:train_validation_index]
  train_label_ids_list = shuffle_label_ids_list[:train_validation_index]
  validation_word_ids_list = shuffle_word_ids_list[train_validation_index:validation_test_index]
  validation_label_ids_list = shuffle_label_ids_list[train_validation_index:validation_test_index]
  test_word_ids_list = shuffle_word_ids_list[validation_test_index:]
  test_label_ids_list = shuffle_label_ids_list[validation_test_index:]

  return train_word_ids_list, train_label_ids_list, \
         validation_word_ids_list, validation_label_ids_list, \
         test_word_ids_list, test_label_ids_list


def shuffle_data(data_path, shuffle_data_path):
  print 'Shuffle data in ' + data_path
  line_list = []
  with open(data_path, mode='r') as data_file:
    for line in data_file:
      line_list.append(line.strip())
  random.shuffle(line_list)
  with open(shuffle_data_path, mode='w') as shuffle_data_file:
    for line in line_list:
      shuffle_data_file.write(line + '\n')


def main():
  raw_data_path = FLAGS.raw_data_path
  segment_path = FLAGS.segment_path
  train_path = FLAGS.train_path
  ids_path = FLAGS.ids_path
  vocab_path = FLAGS.vocab_path
  vocab_size = FLAGS.vocab_size
  train_percent = FLAGS.train_percent
  val_percent = FLAGS.val_percent

  # file_segment(raw_data_path, segment_path)

  words_vocab = create_vocabulary(train_path, os.path.join(vocab_path, 'words_vocab.txt'), vocab_size)
  datasets = prepare_datasets(train_path, ids_path, vocab_path, words_vocab, train_percent, val_percent)


if __name__ == '__main__':
  main()
