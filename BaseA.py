import numpy as np
import tensorflow as tf
import os
import datetime
from DataReader import Reader

class BaseA():
  def __init__(self, FLAGS):
    self.FLAGS = FLAGS   
    
  def formData(self, sentence_tf, aspect_tf):
    with tf.name_scope('Data_formation'):
      sentence_tf_exp = tf.expand_dims(sentence_tf, 2)
      aspect_tf_exp = tf.expand_dims(aspect_tf, 1)
      mask = tf.where(tf.equal(sentence_tf_exp, aspect_tf_exp))
      idx_start_aspect = tf.cast(tf.segment_min(mask[:,1], mask[:,0]), dtype=tf.int64)
      aspect_length = tf.reduce_sum(tf.cast(tf.not_equal(aspect_tf, tf.constant(0, dtype=tf.int64)), dtype=tf.int64), 1)
      idx_end_aspect = idx_start_aspect + aspect_length
      idx_end = tf.reduce_sum(tf.cast(tf.not_equal(sentence_tf, tf.constant(0, dtype=tf.int64)), dtype=tf.int64), 1)

      indices = tf.cast(tf.reshape(tf.tile(tf.range(0, self.FLAGS.N), [tf.shape(sentence_tf)[0]]), [tf.shape(sentence_tf)[0], self.FLAGS.N]),tf.int64)
      idx_lc = tf.tile(tf.expand_dims(idx_start_aspect, 1), [1, self.FLAGS.N])
      idx_a = tf.tile(tf.expand_dims(idx_end_aspect, 1), [1, self.FLAGS.N])
      idx_rc = tf.tile(tf.expand_dims(idx_end, 1), [1, self.FLAGS.N])

      pad_length = self.FLAGS.N - tf.shape(sentence_tf)[1]
      paddings = [tf.zeros([2], dtype=tf.int64), [tf.zeros([1], dtype=tf.int64)[0], pad_length]]
      whole = tf.cast(tf.pad(sentence_tf, paddings, 'CONSTANT', constant_values=0), dtype=tf.int64)

      LC = tf.where(tf.less(indices, idx_lc), whole, tf.zeros_like(whole, dtype=tf.int64))
      A = tf.where(tf.logical_and(tf.greater_equal(indices, idx_lc), tf.less(indices, idx_a)), whole, tf.zeros_like(whole, dtype=tf.int64))
      RC = tf.where(tf.logical_and(tf.greater_equal(indices, idx_a), tf.less(indices, idx_rc)), whole, tf.zeros_like(whole, dtype=tf.int64))
      return LC, A, RC  

  def getMinibatches(self, batch_size, sentence_data, aspect_data, Y):
    with tf.name_scope('Mini_batch'):
      batch_size = tf.cond(batch_size > tf.cast(tf.shape(Y)[0], tf.int64), lambda: tf.cast(tf.shape(Y)[0], tf.int64), lambda: batch_size)
      df = tf.data.Dataset.from_tensor_slices((sentence_data, aspect_data, Y)).repeat().batch(batch_size)
      iterator = df.make_initializable_iterator()
      next_batch = iterator.get_next()
      nr_batches = tf.identity(tf.cast(tf.math.ceil(tf.shape(sentence_data)[0]/tf.cast(batch_size, tf.int32)), tf.int64), name='nr_batches')
      LC, A, RC = self.formData(sentence_data, aspect_data)
      return iterator, next_batch, nr_batches, LC, A, RC  

  def getAccuracy(self, pred, y, data_len, data_len_glove):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.math.argmax(y,1), tf.math.argmax(pred,1)), tf.float64)) * (data_len_glove / data_len)
    return tf.metrics.mean(acc)

  def getLoss(self, logits, y):
      return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y, name='loss')

  def getLossOp(self, loss):
    return tf.metrics.mean(loss, name='loss_op')  

  def getSummaries(self, logits, prediction, Y, data_len, data_len_glove):
    with tf.name_scope('Summaries'):
      loss = self.getLoss(logits, Y)
      loss_log, loss_log_update_op = self.getLossOp(loss)
      loss_scalar = tf.summary.scalar('loss', loss_log)  
      accuracy, accuracy_update_op = self.getAccuracy(prediction, Y, data_len, data_len_glove)
      accuracy_scalar = tf.summary.scalar('accuracy', accuracy)    
      return loss, loss_log, loss_log_update_op, loss_scalar, accuracy, accuracy_update_op, accuracy_scalar

  def initWeights(self):
    with tf.name_scope('weights'):
      weights = {\
       ### Content Attention Module
      'W_1' : tf.get_variable(initializer=tf.random_normal([1, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_1'),
      'W_2' : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_2'),
      'W_3' : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_3'),
      'b_1' : tf.get_variable(initializer=tf.zeros([1, self.d]), dtype=tf.float32, name='b_1'),
       ### MLP
      'W_4' : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_4'),
      'b_2' : tf.get_variable(initializer=tf.zeros([1, self.d]), dtype=tf.float32, name='b_2'),
       ### Linear Layer
      'W_5' : tf.get_variable(initializer=tf.random_normal([self.FLAGS.nr_cat, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_5'),
      'b_3' : tf.get_variable(initializer=tf.zeros([1, self.FLAGS.nr_cat]), dtype=tf.float32, name='b_3')}
      return weights

  def model(self, E, E_A, dropout_prob, weights):
    with tf.name_scope('Content_Attention_Module'):
      with tf.name_scope('scores_of_words'):
        v_a = tf.identity(tf.reduce_sum(E_A, 1) / tf.math.count_nonzero(E_A, 1, dtype=tf.float32), name='v_a')                                                      # (S x d)
        v_a_matrix = tf.tile(tf.expand_dims(v_a, 1), [1, self.FLAGS.N, 1], name='v_a_matrix')                                                                       # (S x N x d)
        C = tf.where(tf.not_equal(E, tf.constant(0.0))[:,:,0], 
                     tf.reshape(tf.tensordot(tf.nn.tanh(
                                                      tf.tensordot(tf.nn.dropout(E, rate=dropout_prob, seed=self.FLAGS.seed), weights['W_2'], [2,1]) +
                                                      tf.tensordot(tf.nn.dropout(v_a_matrix, rate=dropout_prob, seed=self.FLAGS.seed), weights['W_3'], [2,1]) +
                                                      tf.nn.dropout(weights['b_1'], rate=dropout_prob, seed=self.FLAGS.seed)),
                                                    weights['W_1'], [2, 1]),
                                        [-1, self.FLAGS.N]),
                            -1e10*tf.ones([tf.shape(E)[0], self.FLAGS.N]), name='C')                                                                                # (S x N)                          
      with tf.name_scope('attention_weights_for_scores'): 
        A = tf.nn.softmax(C, name='A')                                                                                                                              # (S x N)
      with tf.name_scope('sentence_representation'):
        v_ns = tf.transpose(tf.matmul(tf.transpose(E, [0,2,1]), tf.expand_dims(A, 2)), [0,2,1], name='v_ns')                                                        # (S x 1 x d)
    with tf.name_scope('MLP'):
      v_ms = tf.nn.dropout(tf.nn.tanh(tf.tensordot(tf.nn.dropout(v_ns, rate=dropout_prob, seed=self.FLAGS.seed), weights['W_4'], [2,1]) + 
                                      weights['b_2']), 
                           rate=dropout_prob, seed=self.FLAGS.seed, name='v_ms')                                                                                    # (S x 1 x d)
    with tf.name_scope('Linear_Layer'):
      v_L = tf.reshape(tf.tensordot(v_ms, weights['W_5'], [2,1]) + weights['b_3'], 
                       [-1, self.FLAGS.nr_cat], name='v_L')                                                                                                         # (S x nr_cat)
    return v_L

  def trainModel(self, data, words_dict, train_data_length, train_data_length_glove, data_test=None, test_data_length=None, test_data_length_glove=None, save_to_files=False):
    tf.reset_default_graph()
    sentence_data = tf.placeholder(tf.int64, [None, None], name='sentence_data')
    aspect_data = tf.placeholder(tf.int64, [None, None], name='aspect_data')
    Y = tf.placeholder(tf.int64, [None, 3], name='Y')
    batch_size = tf.placeholder(tf.int64, name='batch_size')
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    dictionary = tf.placeholder(tf.float32, [self.V, self.d], name='dictionary')
    data_len = tf.placeholder(tf.int32, None, name='data_len')
    data_len_glove = tf.placeholder(tf.int32, None, name='data_len_glove')
    
    iterator, next_batch, nr_batches, LC, A, RC = self.getMinibatches(batch_size, sentence_data, aspect_data, Y)
    
    with tf.name_scope('Embeddings'):
      E = tf.nn.embedding_lookup(dictionary, LC+A+RC, name='E')
      E_A = tf.nn.embedding_lookup(dictionary, A, name='E_A')
    
    weights = self.initWeights()
    logits = tf.identity(self.model(E, E_A, dropout_prob, weights), name='logits')

    prediction = tf.nn.softmax(logits, name='prediction')

    loss, loss_log, loss_log_update_op, loss_scalar, accuracy, accuracy_update_op, accuracy_scalar = self.getSummaries(logits, prediction, Y, data_len, data_len_glove)
  
    with tf.name_scope('TrainOp'):
      global_step = tf.Variable(0, trainable=False, name='global_step')  
      optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate)
      train_op = optimizer.minimize(loss, global_step=global_step)      
      
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      if save_to_files:
        writer_train = tf.summary.FileWriter('./Results/logs/{0}/train/{1}'.format(self.__class__.__name__, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')), sess.graph)
        writer_test = tf.summary.FileWriter('./Results/logs/{0}/test/{1}'.format(self.__class__.__name__, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')), sess.graph)
        summary_op = tf.summary.merge_all()   
        saver = tf.train.Saver(max_to_keep = 3, save_relative_paths=True)
      
      sess.run(tf.global_variables_initializer())
      
      batches, _ = sess.run([nr_batches, iterator.initializer], feed_dict={'sentence_data:0':data[0],
                                                                           'aspect_data:0':data[1],
                                                                           'Y:0':data[2],
                                                                           'batch_size:0':self.FLAGS.batch_size})  
      acc_test_best = 0
      loss_test_best = np.inf
      step_best = -1
      for epoch in range(self.FLAGS.num_epochs):
        sess.run(tf.local_variables_initializer())
        for _ in range(batches):
          sentence_batch, aspect_batch, Y_batch = sess.run(next_batch) 
          sess.run(train_op, feed_dict={'sentence_data:0':sentence_batch,
                                        'aspect_data:0':aspect_batch,
                                        'Y:0':Y_batch,
                                        'dropout_prob:0':self.FLAGS.dropout_prob,
                                        'dictionary:0':words_dict})
          sess.run([accuracy_update_op, loss_log_update_op],
                                              feed_dict={'sentence_data:0':sentence_batch,
                                                         'aspect_data:0':aspect_batch,
                                                         'Y:0':Y_batch,
                                                         'dropout_prob:0':self.FLAGS.dropout_prob,
                                                         'dictionary:0':words_dict,
                                                         'data_len:0':train_data_length,
                                                         'data_len_glove:0':train_data_length_glove})          
        acc, cost, step = sess.run([accuracy, loss_log, global_step])

        if save_to_files:
          writer_train.add_summary(sess.run(summary_op), step)
          writer_train.flush()
        if data_test is not None:
          sess.run(tf.local_variables_initializer())
          acc_test, cost_test = sess.run([accuracy_update_op, loss_log_update_op],
                                              feed_dict={'sentence_data:0':data_test[0],
                                                         'aspect_data:0':data_test[1],
                                                         'Y:0':data_test[2],
                                                         'dropout_prob:0':0.0,
                                                         'dictionary:0':words_dict,
                                                         'data_len:0':test_data_length,
                                                         'data_len_glove:0':test_data_length_glove})
          if acc_test > acc_test_best:
            acc_test_best = acc_test
            loss_test_best = cost_test
            step_best  = step
            if save_to_files:
              if not os.path.exists('./Results/ckpt/{}'.format(self.__class__.__name__)):
                os.makedirs('./Results/ckpt/{}'.format(self.__class__.__name__))
              saver.save(sess, './Results/ckpt/{}/model.ckpt'.format(self.__class__.__name__), step)
            improved = '*'
          else:
            improved = ''
          print('Step: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Test Accuracy: {2:>6.1%} Train-Batch Loss: {3} Test Loss: {4} {5}'.
                format(step, acc, acc_test, np.round(cost,5), np.round(cost_test,5), improved)) 
          if save_to_files:
            writer_test.add_summary(sess.run(loss_scalar), step)
            writer_test.add_summary(sess.run(accuracy_scalar), step)  
            writer_test.flush()
        else:
          print('Step: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Train Loss: {2}'.
               format(step, acc, np.round(cost,5))) 
        if step_best + 1000 <= step:
          print('Stopping in step: {}, the best result in step: {}'.format(step, step_best))
          break
      if save_to_files:
        writer_train.close() 
        writer_test.close()
    return acc_test_best, loss_test_best 

  def predict(self, data, words_dict, test_data_length, test_data_length_glove):
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      ckpt = tf.train.latest_checkpoint('./Results/ckpt/{}/'.format(self.__class__.__name__))
      saver = tf.train.import_meta_graph(ckpt+'.meta')

      sess.run(tf.global_variables_initializer())
      saver.restore(sess, ckpt)
      graph = tf.get_default_graph()

      global_step = graph.get_tensor_by_name('TrainOp/global_step:0')
      data_len = graph.get_tensor_by_name('data_len:0')
      data_len_glove = graph.get_tensor_by_name('data_len_glove:0')
      Y = graph.get_tensor_by_name('Y:0')  
      logits = graph.get_tensor_by_name('logits:0')
      prediction = graph.get_tensor_by_name('prediction:0')
      _, accuracy = self.getAccuracy(prediction, Y, data_len, data_len_glove)
      _, loss = self.getLossOp(self.getLoss(logits, Y))
      
      sess.run(tf.local_variables_initializer())
      
      pred, acc, cost, step = sess.run([prediction, accuracy, loss, global_step], 
                                        feed_dict={'sentence_data:0' : data[0],
                                                   'aspect_data:0' : data[1],
                                                   'Y:0' : data[2],
                                                   'dropout_prob:0':0.0,
                                                   'dictionary:0' : words_dict,
                                                   'data_len:0' : test_data_length,
                                                   'data_len_glove:0':test_data_length_glove})
      print('Step: {0:>6}, Accuracy: {1:>6.1%}, Loss: {2}'.
            format(step, acc, np.round(cost,5)))
      return pred 


##########################################

if __name__ == '__main__':
  np.set_printoptions(threshold=np.inf, edgeitems=3, linewidth=120)
  
  FLAGS = tf.app.flags.FLAGS
  for name in list(FLAGS):
    if name not in ('showprefixforinfo',):
      delattr(FLAGS, name)

  tf.app.flags.DEFINE_string('f', '', 'kernel')
  tf.app.flags.DEFINE_boolean('logtostderr', True, 'tensorboard')
  tf.app.flags.DEFINE_float('train_proportion', 1.0, 'Train proportion for train/validation split') #0.75
  tf.app.flags.DEFINE_boolean('shuffle', True, 'Shuffle datasets')
  tf.app.flags.DEFINE_integer('num_epochs', 600, 'Number of iterations')
  tf.app.flags.DEFINE_integer('seed', None, 'Random seed')
  tf.app.flags.DEFINE_float('mean', 0.0, 'Mean of normally initialized variables')
  tf.app.flags.DEFINE_float('stddev', 0.05, 'Standard deviation of normally initialized variables')
  tf.app.flags.DEFINE_float('dropout_prob', 0.3, 'Dropout probability')
  tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Adam optimzier learning rate')
  tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
  tf.app.flags.DEFINE_integer('nr_cat', 3, 'Number of classification categories')  
  tf.app.flags.DEFINE_string('embeddings_path', './Embeddings/glove.42B.300d.txt', 'Embedding path')
  tf.app.flags.DEFINE_boolean('train_model', True, 'Run Ont on train data')
  tf.app.flags.DEFINE_boolean('save_to_files', True, 'Whether to save checkpoints')  
  tf.app.flags.DEFINE_boolean('predict_values', True, 'Run Ont on test data')  
  tf.app.flags.DEFINE_string('train_data_path', './Data/ABSA-16_SB1_Restaurants_Train_Data.xml', 'Train data path')  
  tf.app.flags.DEFINE_string('test_data_path', './Data/ABSA-16_SB1_Restaurants_Test_Gold.xml', 'Test data path')

  reader = Reader(FLAGS)

  embeddings, word2idx, idx2word = reader.readEmbeddings(FLAGS.embeddings_path)

  data_train, longest_sentence_train = reader.readData(FLAGS.train_data_path)
  data_test, longest_sentence_test = reader.readData(FLAGS.test_data_path)
  longest_sentence = max(longest_sentence_train, longest_sentence_test)
  tf.app.flags.DEFINE_integer('N', longest_sentence, 'Length of the longest sentence')

  data_idx_train = reader.transformSent2idx(data_train, word2idx)
  data_idx_test = reader.transformSent2idx(data_test, word2idx)

  model = BaseA(FLAGS)
  
  model.V, model.d = embeddings.shape

  ## Train and evaluate models
  if FLAGS.train_model:
    print('Training...')
    a = datetime.datetime.now()
    model.trainModel(data_idx_train, embeddings, len(data_train), len(data_idx_train[0]), data_idx_test, len(data_test), len(data_idx_test[0]), save_to_files=FLAGS.save_to_files)
    b = datetime.datetime.now()
    print('Time:', b-a)

  ## Predict using trained models    
  if FLAGS.predict_values:
    print('Prediction...')
    pred = model.predict(data_idx_test, embeddings, len(data_test), len(data_idx_test[0]))
  print('')
