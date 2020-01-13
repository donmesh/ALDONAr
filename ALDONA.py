!pip install owlready2
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import os
import datetime
import json
import owlready2 as OWL
from DataReader import Reader

class ALDONA():
  def __init__(self, FLAGS):    
    self.ontology = OWL.get_ontology(FLAGS.ontology_path)
    self.ontology.base_iri = FLAGS.ontology_path
    self.ontology = self.ontology.load()    
    self.polarity_categories = {}
    self.polarity_categories['positive'] = self.ontology.search(iri='*Positive')[0]
    self.polarity_categories['negative'] = self.ontology.search(iri='*Negative')[0]
    self.type1, self.type2, self.type3 = {}, {}, {}    
    classes = set(self.ontology.classes())    
    self.classes_dict = {onto_class: onto_class.lex for onto_class in classes}
    self.classesIntoTypes(classes)
    self.FLAGS = FLAGS
    
  # If unknown words are removed
  def transformSent2idx(self, data, word2idx):
    sentences, aspects, polarities = [], [], []
    UNKNOWN_TOKEN = word2idx['UNK']
    for sentence, aspect, polarity in data:
      temp_aspect = [word2idx.get(word, UNKNOWN_TOKEN) for word in aspect.split()]
      aspect_new = [value for value in temp_aspect if value != UNKNOWN_TOKEN]
      if len(aspect_new) == 0:
        continue
      temp_sentence = [word2idx.get(word, UNKNOWN_TOKEN) for word in sentence.split()]
      sentence_new = [value for value in temp_sentence if value !=  UNKNOWN_TOKEN]
      aspects.append(aspect_new)
      sentences.append(sentence_new)
      polarities.append(self.transformPolarity(polarity))
    sentences = np.array(list(itertools.zip_longest(*sentences, fillvalue=0))).T
    aspects = np.array(list(itertools.zip_longest(*aspects, fillvalue=0))).T
    polarities = np.array(polarities)
    data = [sentences, aspects, polarities]
    return data  

  def transformPolarity(self, polarity):
    if polarity == 'negative': return np.array([1,0,0])
    elif polarity == 'neutral': return np.array([0,1,0])
    elif polarity == 'positive': return np.array([0,0,1])
    else: Exception(polarity)    
    
  def classesIntoTypes(self, classes):
    remove_words = ['property', 'mention', 'positive', 'neutral', 'negative']
    for ont_class in classes:
      class_name = ont_class.__name__.lower()
      if any(word in class_name for word in remove_words): continue
      names = [x.__name__ for x in ont_class.ancestors()]
      names.sort()      
      for name in names:
        if 'Generic' in name:
          self.type1[class_name] = ont_class
          break
        elif any(x in name for x in ['Positive', 'Negative']):
          self.type2[class_name] = ont_class
          break
        elif 'PropertyMention' in name:
          self.type3[class_name] = ont_class
          break    
    
  def getClassPolarity(self, word_lemma_class, negated, type3):
    positive, negative = False, False
    if type3: OWL.sync_reasoner(debug=False) # To set relations of all newly created classes
    if self.polarity_categories['positive'].__subclasscheck__(word_lemma_class):
      if negated:
        positive = False
        negative = True
      else:
        positive = True
    if self.polarity_categories['negative'].__subclasscheck__(word_lemma_class):
      if negated:
        positive = True
        negative = False
      else:
        negative = True
    return positive, negative

  def categoryMatch(self, aspect_class, word_class):
    if aspect_class is None: return False
    aspect_mentions, word_mentions = [], []
    for ancestor in aspect_class.ancestors():
      if 'Mention' in ancestor.__name__:
        aspect_mentions.append(ancestor.__name__.rsplit('Mention',1)[0])
    for ancestor in word_class.ancestors():
      if 'Mention' in ancestor.__name__:
        word_mentions.append(ancestor.__name__.rsplit('Mention',1)[0])
    common = set(aspect_mentions).intersection(set(word_mentions))
    # If they have more than 2 ancestors in common (ontology.Mention, ontology.EntityMention and something else)
    if len(common) > 2: return True 
    else: return False

  def addSubclass(self, word_class, aspect_class):
    class_name = word_class.__name__ + aspect_class.__name__
    new_class = OWL.types.new_class(class_name, (word_class, aspect_class))
    self.type3[new_class.__name__.lower()] = new_class
    return new_class 

  def isNegated(self, word, words_in_sentence):
    negation = ({"not","no","never","isnt","arent","wont","wasnt","werent", 
                 "havent","hasnt", "nt", "cant", "couldnt", "dont", "doesnt"})
    negated = False
    index = words_in_sentence.index(word)
    check = set(words_in_sentence[max(index-3,0):index])
    if check.intersection(negation): negated = True
    return negated

  def predictSentiment(self, sentence, aspect):
    lemmatizer = nltk.WordNetLemmatizer()
    positive_list, negative_list = [], []
    sentence_classes = {}
    words_in_sentence = sentence.split() 
    aspect_class = None
    
    for word, tag in np.array(nltk.pos_tag(nltk.word_tokenize(aspect))):
      if tag.startswith("V"): aspect_lemma = lemmatizer.lemmatize(word, "v")   # Verb
      elif tag.startswith("J"): aspect_lemma = lemmatizer.lemmatize(word, "a") # Adjective
      elif tag.startswith("R"): aspect_lemma = lemmatizer.lemmatize(word, "r") # Adverb
      else: aspect_lemma = lemmatizer.lemmatize(word)                          # Other words do not change
      for ont_class in list(self.classes_dict.values()):
        if aspect_lemma in ont_class:
          aspect_class = list(self.classes_dict.keys())[list(self.classes_dict.values()).index(ont_class)]
    
    for word, tag in np.array(nltk.pos_tag(nltk.word_tokenize(sentence))):
      if tag.startswith("V"): word_lemma = lemmatizer.lemmatize(word, "v")   # Verb
      elif tag.startswith("J"): word_lemma = lemmatizer.lemmatize(word, "a") # Adjective
      elif tag.startswith("R"): word_lemma = lemmatizer.lemmatize(word, "r") # Adverb
      else: word_lemma = lemmatizer.lemmatize(word)                          # Other words do not change
      for ont_class in list(self.classes_dict.values()):
        if word_lemma in ont_class:
          word_class = list(self.classes_dict.keys())[list(self.classes_dict.values()).index(ont_class)]
          sentence_classes[word] = word_class
          if word == aspect:
            aspect_class = word_class
          is_negated = self.isNegated(word, words_in_sentence)
          if word_lemma in self.type1:
            positive, negative = self.getClassPolarity(word_class, is_negated, False)
            positive_list.append(positive)
            negative_list.append(negative)
          elif word_lemma in self.type2:
            if self.categoryMatch(aspect_class, word_class):
              positive, negative = self.getClassPolarity(word_class, is_negated, False)
              positive_list.append(positive)
              negative_list.append(negative)              
          elif word_lemma in self.type3:
            if (aspect_class != word_class) and (aspect_class is not None):
              new_class = self.addSubclass(word_class, aspect_class)
              positive, negative = self.getClassPolarity(new_class, is_negated, True)
              positive_list.append(positive)
              negative_list.append(negative) 
    if (True in positive_list) and (True not in negative_list):
      prediction = np.array([[0,0,1]])
    elif (True not in positive_list) and (True in negative_list):
      prediction = np.array([[1,0,0]])
    else:
      prediction = None
    return prediction

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
       ### Bidirectional LSTM
      'W_fw_L' : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_fw_L'),
      'W_bw_L' : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_bw_L'),
      'b_bi_L' : tf.get_variable(initializer=tf.zeros([1, self.d]),dtype=tf.float32, name='b_bi_L'),
      'W_fw_R' : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_fw_R'),
      'W_bw_R' : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_bw_R'),
      'b_bi_R' : tf.get_variable(initializer=tf.zeros([1, self.d]), dtype=tf.float32, name='b_bi_R'),                 
       ### MLP Beta
      'W_1' : tf.get_variable(initializer=tf.random_normal([1, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_1'),
      'b_1' : tf.get_variable(initializer=tf.zeros([1,1]), dtype=tf.float32, name='b_1'),
      'b_l' : tf.get_variable(initializer=0.5, trainable=False, dtype=tf.float32, name='b_l'),
      'W_2' : tf.get_variable(initializer=tf.random_normal([1, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_2'),
      'b_2' : tf.get_variable(initializer=tf.zeros([1,1]), dtype=tf.float32, name='b_2'),
      'b_r' : tf.get_variable(initializer=0.5, trainable=False, dtype=tf.float32, name='b_r'),  
       ### Sentence Level Content Attention Module                      
      'W_3' : tf.get_variable(initializer=tf.random_normal([1, self.FLAGS.m], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_3'),
      'W_4' : tf.get_variable(initializer=tf.random_normal([self.FLAGS.m, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_4'),
      'W_5' : tf.get_variable(initializer=tf.random_normal([self.FLAGS.m, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_5'),
      'W_6' : tf.get_variable(initializer=tf.random_normal([self.FLAGS.m, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_6'),                      
      'b_3' : tf.get_variable(initializer=tf.zeros([1, self.FLAGS.m]), dtype=tf.float32, name='b_3'),
       ### Classification Module
      'W_7'  : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_7'),
      'W_8'  : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_8'),
      'b_4'  : tf.get_variable(initializer=tf.zeros([1, self.d]), dtype=tf.float32, name='b_4'),
      'W_9'  : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_9'),
      'W_10' : tf.get_variable(initializer=tf.random_normal([self.d, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_10'),
      'b_5'  : tf.get_variable(initializer=tf.zeros([1, self.d]), dtype=tf.float32, name='b_5'),
      'W_11' : tf.get_variable(initializer=tf.random_normal([self.FLAGS.k, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_11'),
      'W_12' : tf.get_variable(initializer=tf.random_normal([self.FLAGS.k, self.d], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_12'),
      'b_6'  : tf.get_variable(initializer=tf.zeros([1, self.FLAGS.k]), dtype=tf.float32, name='b_6'),
       ### Linear Layer
      'W_13' : tf.get_variable(initializer=tf.random_normal([self.FLAGS.nr_cat, self.FLAGS.k], mean=self.FLAGS.mean, stddev=self.FLAGS.stddev, seed=self.FLAGS.seed), dtype=tf.float32, name='W_13'),
      'b_7'  : tf.get_variable(initializer=tf.zeros([1, self.FLAGS.nr_cat]), dtype=tf.float32, name='b_7')} 
      return weights

  def getBeta(self, H_LS_masked, H_RS_masked, mask_left, mask_aspect, mask_right, mask_left_with_aspect, mask_right_with_aspect, weights):    
      with tf.name_scope('MLP_beta'):
        left = tf.reshape(mask_left[:,:,0], [-1, self.FLAGS.N, 1])
        aspect = tf.reshape(mask_aspect[:,:,0], [-1, self.FLAGS.N, 1])
        right = tf.reshape(mask_right[:,:,0], [-1, self.FLAGS.N, 1])
        left_with_aspect = tf.reshape(mask_left_with_aspect[:,:,0], [-1, self.FLAGS.N, 1])
        right_with_aspect = tf.reshape(mask_right_with_aspect[:,:,0], [-1, self.FLAGS.N, 1])
  
        beta_LS = tf.identity(tf.nn.sigmoid(tf.where(left_with_aspect,
                                                     tf.tensordot(H_LS_masked, weights['W_1'], [2, 1]) + weights['b_1'], 
                                                     -1e10*tf.ones_like(left_with_aspect, dtype=tf.float32))) + 
                              weights['b_l'], name='beta_LS')                                                                              # (S x N x 1)
        beta_RS = tf.identity(tf.nn.sigmoid(tf.where(right_with_aspect, 
                                                     tf.tensordot(H_RS_masked, weights['W_2'], [2, 1]) + weights['b_2'], 
                                                     -1e10*tf.ones_like(right_with_aspect, dtype=tf.float32))) + 
                              weights['b_r'], name='beta_RS')                                                                              # (S x N x 1)
        beta_LC = tf.where(left, beta_LS, tf.zeros_like(left, dtype=tf.float32), name='beta_LC')                                           # (S x N x 1)
        beta_A  = tf.where(aspect, 
                           (tf.where(aspect, beta_LS, tf.zeros_like(aspect, dtype=tf.float32)) + 
                            tf.where(aspect, beta_RS, tf.zeros_like(aspect, dtype=tf.float32))) / 2,
                           tf.zeros_like(aspect, dtype=tf.float32), name='beta_A')                                                         # (S x N x 1)
        beta_RC = tf.where(right, beta_RS, tf.zeros_like(right, dtype=tf.float32), name='beta_RC')                                         # (S x N x 1)
        beta = tf.identity(tf.reshape(beta_LC + beta_A + beta_RC, [-1, self.FLAGS.N]), name='beta')                                        # (S x N)
        return beta

  def getBGRU(self, side, E, length, sentence_length, mask_left_with_aspect, mask_right_with_aspect, dropout_prob, weights):
    with tf.name_scope('BGRU'):
      W_fw = 'W_fw_L' if side == 'left' else 'W_fw_R'
      W_bw = 'W_bw_L' if side == 'left' else 'W_bw_R'
      b_bi = 'b_bi_L' if side == 'left' else 'b_bi_R'
      with tf.variable_scope(side) as scope:
        cell = tf.nn.rnn_cell.LSTMCell(self.d)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-dropout_prob, input_keep_prob=1-dropout_prob,
                                             state_keep_prob=1-dropout_prob, seed=self.FLAGS.seed, dtype=tf.float32)
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, E, sequence_length=length, dtype=tf.float32, scope=scope)   
        H_fw, H_bw = output                                                                                                                # (S x N x d) 

        H = tf.nn.tanh(tf.tensordot(H_fw, weights[W_fw], [2, 1]) +
                       tf.tensordot(H_bw, weights[W_bw], [2, 1]) +
                       weights[b_bi])                                                                                                      # (S x N x d)
  
        if side == 'left':
          H_masked = tf.where(mask_left_with_aspect, 
                              tf.reverse_sequence(H, length, seq_axis=1, batch_axis=0), 
                              tf.zeros_like(mask_left_with_aspect, dtype=tf.float32))                                                      # (S x N x d)
        else:
          H_masked = tf.where(mask_right_with_aspect, 
                              tf.reverse_sequence(tf.reverse_sequence(H, length, seq_axis=1, batch_axis=0), 
                                                  sentence_length, seq_axis=1, batch_axis=0), 
                              tf.zeros_like(mask_right_with_aspect, dtype=tf.float32))                                                     # (S x N x d)
        return H_masked

  def model(self, E, E_LC, E_A, E_RC, dropout_prob, weights):
    with tf.name_scope('Context_Attention_Module'):
      mask_left = tf.not_equal(E_LC, tf.constant(0.0), name='mask_left')
      mask_left_with_aspect = tf.not_equal(E_LC+E_A, tf.constant(0.0), name='mask_left_with_aspect')
      mask_aspect = tf.not_equal(E_A, tf.constant(0.0), name='mask_aspect')
      mask_right = tf.not_equal(E_RC, tf.constant(0.0), name='mask_right')
      mask_right_with_aspect = tf.not_equal(E_RC+E_A, tf.constant(0.0), name='mask_right_with_aspect')
      mask_sentence = tf.not_equal(E, tf.constant(0.0), name='mask_sentence')
      
      left_length = tf.reduce_sum(tf.cast(mask_left_with_aspect[:,:,0], tf.int32), 1, name='left_length')                                                                  # (S)
      right_length = tf.reduce_sum(tf.cast(mask_right_with_aspect[:,:,0], tf.int32), 1, name='right_length')                                                               # (S)
      sentence_length = tf.reduce_sum(tf.cast(mask_sentence[:,:,0], tf.int32), 1, name='sentence_length')                                                                  # (S)
      
      E_LS_r = tf.reverse_sequence(E_LC+E_A, left_length, seq_axis=1, batch_axis=0)                                                                                        # (S x N x d)
      E_RS_r = tf.reverse_sequence(tf.reverse_sequence(E_RC+E_A, sentence_length, seq_axis=1, batch_axis=0), right_length, seq_axis=1, batch_axis=0)                       # (S x N x d)
      
      H_LS = tf.identity(self.getBGRU('left', E_LS_r, left_length, sentence_length, mask_left_with_aspect, mask_right_with_aspect, dropout_prob, weights), name='H_LS')   # (S x N x d)
      H_RS = tf.identity(self.getBGRU('right', E_RS_r, right_length, sentence_length, mask_left_with_aspect, mask_right_with_aspect, dropout_prob, weights), name='H_RS') # (S x N x d)

      beta = self.getBeta(H_LS, H_RS, mask_left, mask_aspect, mask_right, mask_left_with_aspect, mask_right_with_aspect, weights)                                          # (S x N)

    with tf.name_scope('context_attention_weighted_memory'):
      M_w = tf.identity(tf.tile(tf.expand_dims(beta, 2),[1,1,self.d]) * E, name='M_w') 

    with tf.name_scope('Sentence_Level_Content_Attention_Module'):
      with tf.name_scope('scores_of_words'):
        v_a = tf.identity(tf.reduce_sum(E_A, 1) / tf.math.count_nonzero(E_A, 1, dtype=tf.float32), name='v_a')                                                             # (S x d)
        v_s = tf.identity(tf.reduce_sum(E, 1) / tf.math.count_nonzero(E, 1, dtype=tf.float32), name='v_s')                                                                 # (S x d)
        v_a_matrix = tf.tile(tf.expand_dims(v_a, 1), [1, self.FLAGS.N, 1], name='v_a_matrix')                                                                              # (S x N x d)
        v_s_matrix = tf.tile(tf.expand_dims(v_s, 1), [1, self.FLAGS.N, 1], name='v_s_matrix')                                                                              # (S x N x d)                           
        C = tf.where(tf.not_equal(E, tf.constant(0.0))[:,:,0], 
                     tf.reshape(tf.tensordot(tf.nn.tanh(
                                                     tf.tensordot(tf.nn.dropout(M_w, rate=dropout_prob, seed=self.FLAGS.seed), weights['W_4'], [2,1]) +
                                                     tf.tensordot(tf.nn.dropout(v_a_matrix, rate=dropout_prob, seed=self.FLAGS.seed), weights['W_5'], [2,1]) +
                                                     tf.tensordot(tf.nn.dropout(v_s_matrix, rate=dropout_prob, seed=self.FLAGS.seed), weights['W_6'], [2,1]) +
                                                     tf.nn.dropout(weights['b_3'], rate=dropout_prob, seed=self.FLAGS.seed)),
                                             weights['W_3'], [2, 1]),
                                [-1, self.FLAGS.N]),
                     -1e10*tf.ones([tf.shape(E)[0], self.FLAGS.N]), name='C')                                                                                              # (S x N)

      with tf.name_scope('attention_weights_for_scores'): 
        A = tf.nn.softmax(C, name='A')   
      with tf.name_scope('weighted_embedding_vector'):
        v_we = tf.transpose(tf.matmul(tf.transpose(M_w, [0,2,1]), tf.expand_dims(A, 2)), [0,2,1])                                                                          # (S x 1 x d)

    with tf.name_scope('Classification_Module'):
      v_aw = tf.nn.dropout(tf.nn.tanh(
                                  tf.tensordot(tf.nn.dropout(tf.reshape(v_s, [-1, 1, self.d]), rate=dropout_prob, seed=self.FLAGS.seed), weights['W_7'], [2,1]) + 
                                  tf.tensordot(tf.nn.dropout(v_we, rate=dropout_prob, seed=self.FLAGS.seed), weights['W_8'], [2,1]) +
                                  weights['b_4']),
                           rate=dropout_prob, seed=self.FLAGS.seed, name='v_aw')                                                                                           # (S x 1 x d)

      v_sw = tf.nn.dropout(tf.nn.tanh(
                                  tf.tensordot(tf.nn.dropout(tf.reshape(v_a, [-1, 1, self.d]), rate=dropout_prob, seed=self.FLAGS.seed), weights['W_9'], [2,1]) + 
                                  tf.tensordot(tf.nn.dropout(v_we, rate=dropout_prob, seed=self.FLAGS.seed), weights['W_10'], [2,1]) +
                                  weights['b_5']),
                           rate=dropout_prob, seed=self.FLAGS.seed, name='v_sw')                                                                                           # (S x 1 x d)

      v_o = tf.nn.dropout(tf.nn.tanh(tf.tensordot(v_aw, weights['W_11'], [2,1]) +
                                     tf.tensordot(v_sw, weights['W_12'], [2,1]) +
                                     weights['b_6']),
                          rate=dropout_prob, seed=self.FLAGS.seed, name='v_o')
    with tf.name_scope('Linear_Layer'):
      v_L = tf.reshape(tf.tensordot(v_o, weights['W_13'], [2,1]) + weights['b_7'], 
                       [-1, self.FLAGS.nr_cat], name='v_L')                                                                                                                 # (S x nr_cat)
    return v_L

  def trainModel(self, data, words_dict, data_test=None, save_to_files=False):
    predictions_ont = np.zeros([1,3])
    data_DBGRU = []
    data_ont = []

    data_idx = self.transformSent2idx(data, word2idx)
    train_data_length = len(data)
    train_data_length_glove = len(data_idx[0])

    for sentence, aspect, polarity in data:
      prediction = self.predictSentiment(sentence, aspect)
      if prediction is not None:
        predictions_ont = np.concatenate([predictions_ont, prediction])
        data_ont.append([sentence, aspect, polarity])
      else:
        data_DBGRU.append([sentence, aspect, polarity])

    predictions_ont = predictions_ont[1:,:]
    y_ont = np.array(list(map(self.transformPolarity, [y for _, _ , y in data_ont])))   
    
    data_DBGRU_idx = self.transformSent2idx(data_DBGRU, word2idx)

    if data_test is not None:
      predictions_ont_test = np.zeros([1,3])
      data_DBGRU_test = []
      data_ont_test = []

      data_idx_test = self.transformSent2idx(data_test, word2idx)
      test_data_length = len(data_test)
      test_data_length_glove = len(data_idx_test[0])  

      for sentence, aspect, polarity in data_test:
        prediction_test = self.predictSentiment(sentence, aspect)
        if prediction_test is not None:
          predictions_ont_test = np.concatenate([predictions_ont_test, prediction_test])
          data_ont_test.append([sentence, aspect, polarity])
        else:
          data_DBGRU_test.append([sentence, aspect, polarity])
      predictions_ont_test = predictions_ont_test[1:,:]
      
      y_ont_test = np.array(list(map(self.transformPolarity, [y for _, _ , y in data_ont_test])))
    
      data_DBGRU_idx_test = self.transformSent2idx(data_DBGRU_test, word2idx)


    if len(data_DBGRU_idx[0]) > 0 or len(data_DBGRU_idx_test[0]) > 0:
      tf.reset_default_graph()
      sentence_data = tf.placeholder(tf.int64, [None, None], name='sentence_data')
      aspect_data = tf.placeholder(tf.int64, [None, None], name='aspect_data')
      Y = tf.placeholder(tf.int64, [None, 3], name='Y')
      Y_ont = tf.placeholder(tf.int64, [None, 3], name='Y_ont')
      prediction_ont = tf.placeholder(tf.float32, [None, 3], name='prediction_ont')
      batch_size = tf.placeholder(tf.int64, name='batch_size')
      dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
      dictionary = tf.placeholder(tf.float32, [self.V, self.d], name='dictionary')    
      data_len = tf.placeholder(tf.int32, None, name='data_len')
      data_len_glove = tf.placeholder(tf.int32, None, name='data_len_glove')

      iterator, next_batch, nr_batches, LC, A, RC = self.getMinibatches(batch_size, sentence_data, aspect_data, Y)

      with tf.name_scope('Embeddings'):
        E = tf.nn.embedding_lookup(dictionary, LC+A+RC, name='E')
        E_LC = tf.nn.embedding_lookup(dictionary, LC, name='E_LC')
        E_A = tf.nn.embedding_lookup(dictionary, A, name='E_A')
        E_RC = tf.nn.embedding_lookup(dictionary, RC, name='E_RC')
      
      weights = self.initWeights()

      logits = tf.identity(self.model(E, E_LC, E_A, E_RC, dropout_prob, weights), name='logits')

      prediction = tf.nn.softmax(logits, name='prediction')

      loss = self.getLoss(logits, Y)

      with tf.name_scope('TrainOp'):
        global_step = tf.Variable(0, trainable=False, name='global_step')  
        optimizer = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

      # Checking whether Ont has classified anything
      logits_stacked = tf.identity(tf.cond(tf.not_equal(tf.reduce_sum(Y_ont), tf.constant(-3, dtype=tf.int64)),
                                            lambda: tf.concat([prediction_ont, logits], axis=0),
                                            lambda: logits),
                                    name='logits_stacked')
      prediction_stacked = tf.identity(tf.cond(tf.not_equal(tf.reduce_sum(Y_ont), tf.constant(-3, dtype=tf.int64)),
                                                lambda: tf.concat([prediction_ont, prediction], axis=0),
                                                lambda: prediction), 
                                        name='prediction_stacked')
      
      Y_stacked = tf.identity(tf.cond(tf.not_equal(tf.reduce_sum(Y_ont), tf.constant(-3, dtype=tf.int64)),
                                      lambda: tf.concat([Y_ont, Y], axis=0),
                                      lambda: Y),
                              name='Y_stacked')

      _, loss_log, loss_log_update_op, loss_scalar, accuracy, \
        accuracy_update_op, accuracy_scalar = self.getSummaries(logits_stacked, prediction_stacked, Y_stacked, data_len, data_len_glove)

      with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        if save_to_files:
          writer_train = tf.summary.FileWriter('./Results/logs/{0}/train/{1}'.format(self.__class__.__name__, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')), sess.graph)
          writer_test = tf.summary.FileWriter('./Results/logs/{0}/test/{1}'.format(self.__class__.__name__, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')), sess.graph)
          summary_op = tf.summary.merge_all()   
          saver = tf.train.Saver(max_to_keep = 3, save_relative_paths=True)    

        sess.run(tf.global_variables_initializer())    

        batches, _ = sess.run([nr_batches, iterator.initializer], feed_dict={'sentence_data:0':data_idx[0],
                                                                             'aspect_data:0':data_idx[1],
                                                                             'Y:0':data_idx[2],
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
                       
          step = sess.run(global_step)

          if len(data_DBGRU_idx[0]) > 0:            
            sess.run([accuracy_update_op, loss_log_update_op],
                                                  feed_dict={'sentence_data:0':data_DBGRU_idx[0],
                                                             'aspect_data:0':data_DBGRU_idx[1],
                                                             'Y:0':data_DBGRU_idx[2],
                                                             'Y_ont:0':y_ont if len(y_ont)>0 else np.array([[-1,-1,-1]]),
                                                             'prediction_ont:0':predictions_ont if len(predictions_ont)>0 else np.array([[-1,-1,-1]]),
                                                             'dropout_prob:0':self.FLAGS.dropout_prob,
                                                             'dictionary:0':words_dict,
                                                             'data_len:0':train_data_length,
                                                             'data_len_glove:0':train_data_length_glove})              

            acc, cost = sess.run([accuracy, loss_log])    
            if save_to_files:
              writer_train.add_summary(sess.run(summary_op), step)
              writer_train.flush()  
          else:
            acc = sklearn.metrics.accuracy_score(np.argmax(predictions_ont, 1), np.argmax(y_ont, 1))
            cost = sklearn.metrics.log_loss(y_ont, predictions_ont)

          if len(data_DBGRU_idx_test[0]) > 0:
            sess.run(tf.local_variables_initializer())
            sess.run([accuracy_update_op, loss_log_update_op],
                                                  feed_dict={'sentence_data:0':data_DBGRU_idx_test[0],
                                                             'aspect_data:0':data_DBGRU_idx_test[1],
                                                             'Y:0':data_DBGRU_idx_test[2],
                                                             'Y_ont:0':y_ont_test if len(y_ont_test)>0 else np.array([[-1,-1,-1]]),
                                                             'prediction_ont:0':predictions_ont_test if len(predictions_ont_test)>0 else np.array([[-1,-1,-1]]),
                                                             'dropout_prob:0':0.0,
                                                             'dictionary:0':words_dict,
                                                             'data_len:0':test_data_length,
                                                             'data_len_glove:0':test_data_length_glove})
            acc_test, cost_test = sess.run([accuracy, loss_log])
            if save_to_files:
              writer_test.add_summary(sess.run(loss_scalar), step)
              writer_test.add_summary(sess.run(accuracy_scalar), step)  
              writer_test.flush()            
          else:
            acc_test = sklearn.metrics.accuracy_score(np.argmax(predictions_ont_test, 1), np.argmax(y_ont_test, 1))
            cost_test = sklearn.metrics.log_loss(y_ont_test, predictions_ont_test) 

          if acc_test > acc_test_best:
            acc_test_best = acc_test
            loss_test_best = cost_test
            epoch_best  = epoch
            if save_to_files:
              if not os.path.exists('./Results/ckpt/{}'.format(self.__class__.__name__)):
                os.makedirs('./Results/ckpt/{}'.format(self.__class__.__name__))
              saver.save(sess, './Results/ckpt/{}/model.ckpt'.format(self.__class__.__name__), step)
            improved = '*'
          else:
            improved = ''

          print('Step: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Test Accuracy: {2:>6.1%} Train-Batch Loss: {3} Test Loss: {4} {5}'.
                format(step, acc, acc_test, np.round(cost,5), np.round(cost_test,5), improved)) 
          if epoch_best + 1000 <= epoch:
            print('Stopping in step: {}, the best result in epoch: {}'.format(step, epoch_best))
            break             
        if save_to_files:
          writer_train.close() 
          writer_test.close() 
    else:
      acc = sklearn.metrics.accuracy_score(np.argmax(predictions_ont, 1), np.argmax(y_ont, 1))
      cost = sklearn.metrics.log_loss(y_ont, predictions_ont)
      if data_test is not None:
        acc_test_best = sklearn.metrics.accuracy_score(np.argmax(predictions_ont_test, 1), np.argmax(y_ont_test, 1))
        cost_test_best = sklearn.metrics.log_loss(y_ont_test, predictions_ont_test)
        print('Train Accuracy: {0:>6.1%}, Test Accuracy: {1:>6.1%} Train Loss: {2} Test Loss: {3}'.
              format(acc, acc_test_best, np.round(cost,5), np.round(cost_test_best,5)))         
      else:
        print('Train Accuracy: {0:>6.1%}, Train Loss: {1}'.format(acc, np.round(cost,5)))     
    return acc_test_best, loss_test_best

  def predict(self, data, words_dict, word2idx):
    predictions_ont = np.zeros([1,3])
    data_DBGRU = []
    data_ont = []

    data_idx = self.transformSent2idx(data, word2idx)
    data_length = len(data)
    data_length_glove = len(data_idx[0])  

    for sentence, aspect, polarity in data:
      prediction = self.predictSentiment(sentence, aspect)
      if prediction is not None:
        predictions_ont = np.concatenate([predictions_ont, prediction])
        data_ont.append([sentence, aspect, polarity])
      else:
        data_DBGRU.append([sentence, aspect, polarity])

    predictions_ont = predictions_ont[1:,:]
    y_ont = np.array(list(map(self.transformPolarity, [y for _, _ , y in data_ont])))   

    data_DBGRU_idx = self.transformSent2idx(data_DBGRU, word2idx)

    if len(data_DBGRU_idx[0]) > 0:
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
        Y = graph.get_tensor_by_name('Y_stacked:0')  
        logits = graph.get_tensor_by_name('logits_stacked:0')
        prediction = graph.get_tensor_by_name('prediction_stacked:0')
        _, accuracy = self.getAccuracy(prediction, Y, data_len, data_len_glove)
        _, loss = self.getLossOp(self.getLoss(logits, Y)) 

        sess.run(tf.local_variables_initializer())    
       
        predictions, acc, cost, step = sess.run([prediction, accuracy, loss, global_step], 
                                                 feed_dict={'sentence_data:0':data_DBGRU_idx[0],
                                                             'aspect_data:0':data_DBGRU_idx[1],
                                                             'Y:0':data_DBGRU_idx[2],
                                                             'Y_ont:0':y_ont if len(y_ont)>0 else np.array([[-1,-1,-1]]),
                                                             'prediction_ont:0':predictions_ont if len(predictions_ont)>0 else np.array([[-1,-1,-1]]),
                                                             'dropout_prob:0':0.0,
                                                             'dictionary:0':words_dict,
                                                             'data_len:0':data_length,
                                                             'data_len_glove:0':data_length_glove})        
        print('Step: {0:>6}, Accuracy: {1:>6.1%}, Loss: {2}'.
              format(step, acc, np.round(cost,5)))   
        return predictions 
    else:
      acc = sklearn.metrics.accuracy_score(np.argmax(predictions_ont, 1), np.argmax(y_ont, 1))
      cost = sklearn.metrics.log_loss(y_ont, predictions_ont)
      print('Accuracy: {0:>6.1%}, Loss: {1}'.format(acc, np.round(cost,5)))        
      return predictions_ont       


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
  tf.app.flags.DEFINE_integer('m', 300, 'm')
  tf.app.flags.DEFINE_integer('k', 150, 'k')
  tf.app.flags.DEFINE_integer('nr_cat', 3, 'Number of classification categories')  
  tf.app.flags.DEFINE_string('embeddings_path', './Embeddings/glove.42B.300d.txt', 'Embedding path')
  tf.app.flags.DEFINE_boolean('train_model', True, 'Run Ont on train data')
  tf.app.flags.DEFINE_boolean('save_to_files', True, 'Whether to save checkpoints')  
  tf.app.flags.DEFINE_boolean('predict_values', True, 'Run Ont on test data')
  tf.app.flags.DEFINE_string('ontology_path', './Ontology/Ontology_restaurants.owl', 'Ontology path')
  tf.app.flags.DEFINE_string('train_data_path', './Data/ABSA-16_SB1_Restaurants_Train_Data.xml', 'Train data path')  
  tf.app.flags.DEFINE_string('test_data_path', './Data/ABSA-16_SB1_Restaurants_Test_Gold.xml', 'Test data path')

  reader = Reader(FLAGS)

  embeddings, word2idx, idx2word = reader.readEmbeddings(FLAGS.embeddings_path)

  data_train, longest_sentence_train = reader.readData(FLAGS.train_data_path)
  data_test, longest_sentence_test = reader.readData(FLAGS.test_data_path)
  longest_sentence = max(longest_sentence_train, longest_sentence_test)
  tf.app.flags.DEFINE_integer('N', longest_sentence, 'Length of the longest sentence')

  model = ALDONA(FLAGS)
  
  model.V, model.d = embeddings.shape
  model.longest_sentence = longest_sentence

  ## Train and evaluate models
  if FLAGS.train_model:
    print('Training...')
    a = datetime.datetime.now()
    model.trainModel(data_train, embeddings, data_test, save_to_files=FLAGS.save_to_files)
    b = datetime.datetime.now()
    print('Time:', b-a)

  ## Predict using trained models    
  if FLAGS.predict_values:
    print('Prediction...')
    pred = model.predict(data_test, embeddings, word2idx)
  print('')
