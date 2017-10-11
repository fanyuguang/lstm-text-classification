#!/usr/bin/env Python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import data_utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tfrecords_path', 'data/tfrecords/', 'tfrecords directory')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'words batch size')
tf.app.flags.DEFINE_integer('min_after_dequeue', 10000, 'min after dequeue')
tf.app.flags.DEFINE_integer('num_threads', 1, 'read batch num threads')
tf.app.flags.DEFINE_integer('num_steps', 50, 'num steps, equals the length of words')


def create_record(word_datasets, label_datasets, tfrecords_path):
  print 'Create record to ' + tfrecords_path
  writer = tf.python_io.TFRecordWriter(tfrecords_path)
  for (word_ids, label_ids) in zip(word_datasets, label_datasets):
    word_list = [int(word) for word in word_ids.strip().split() if word]
    label = [int(label_ids)]
    example = tf.train.Example(features=tf.train.Features(feature={
      'words': tf.train.Feature(int64_list=tf.train.Int64List(value=word_list)),
      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
    }))
    writer.write(example.SerializeToString())
  writer.close()


def read_and_decode(tfrecords_path):
  print 'Read record from ' + tfrecords_path
  filename_queue = tf.train.string_input_producer([tfrecords_path], num_epochs=None)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example, features={
    # 'words': tf.FixedLenFeature([50], tf.int64),
    'words': tf.VarLenFeature(tf.int64),
    'label': tf.FixedLenFeature([], tf.int64),
  })
  num_steps = FLAGS.num_steps
  words = features['words']
  words = tf.sparse_to_dense(sparse_indices=words.indices[:num_steps], output_shape=[num_steps],
                             sparse_values=words.values[:num_steps], default_value=0)
  label = features['label']
  batch_size = FLAGS.batch_size
  min_after_dequeue = FLAGS.min_after_dequeue
  capacity = min_after_dequeue + 3 * batch_size
  num_threads = FLAGS.num_threads
  words_batch, label_batch = tf.train.shuffle_batch([words, label], batch_size=batch_size, capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue, num_threads=num_threads)
  return words_batch, label_batch


def print_all(tfrecords_path):
  number = 1
  for serialized_example in tf.python_io.tf_record_iterator(tfrecords_path):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    words = example.features.feature['words'].int64_list.value
    labels = example.features.feature['label'].int64_list.value
    word_list = [word for word in words]
    labels = [label for label in labels]
    print('Number:{}, label: {}, features: {}'.format(number, labels, word_list))
    number += 1


def print_shuffle(tfrecords_path):
  words_batch, label_batch = read_and_decode(tfrecords_path)
  with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():
        batch_words_r, batch_label_r = sess.run([words_batch, label_batch])
        print 'batch_words_r : ',
        print batch_words_r.shape
        print batch_words_r
        print 'batch_label_r : ',
        print batch_label_r.shape
        print batch_label_r
    except tf.errors.OutOfRangeError:
      print 'Done reading'
    finally:
      coord.request_stop()
    coord.join(threads)


def main(_):
  train_path = FLAGS.train_path
  ids_path = FLAGS.ids_path
  vocab_path = FLAGS.vocab_path
  vocab_size = FLAGS.vocab_size
  tfrecords_path = FLAGS.tfrecords_path
  train_percent = FLAGS.train_percent
  val_percent = FLAGS.val_percent

  words_vocab = data_utils.create_vocabulary(train_path, os.path.join(vocab_path, 'words_vocab.txt'), vocab_size)
  datasets = data_utils.prepare_datasets(train_path, ids_path, vocab_path, words_vocab, train_percent, val_percent)
  train_word_ids_list, train_label_ids_list, validation_word_ids_list, validation_label_ids_list, \
  test_word_ids_list, test_label_ids_list = datasets

  create_record(train_word_ids_list, train_label_ids_list, os.path.join(tfrecords_path, 'train.tfrecords'))
  create_record(validation_word_ids_list, validation_label_ids_list, os.path.join(tfrecords_path, 'validate.tfrecords'))
  create_record(test_word_ids_list, test_label_ids_list, os.path.join(tfrecords_path, 'test.tfrecords'))

  print_all(os.path.join(tfrecords_path, 'test.tfrecords'))
  # print_shuffle(os.path.join(tfrecords_path, 'test.tfrecords'))


if __name__ == '__main__':
  tf.app.run()
