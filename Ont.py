import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

import owlready2 as OWL
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import os
import datetime
import json
from DataReader import Reader


class Ont():
  def __init__(self, ontology):    
    self.ontology = OWL.get_ontology(ontology)
    self.ontology.base_iri = ontology
    self.ontology = self.ontology.load()    
    self.polarity_categories = {}
    self.polarity_categories['positive'] = self.ontology.search(iri='*Positive')[0]
    self.polarity_categories['negative'] = self.ontology.search(iri='*Negative')[0]
    self.type1, self.type2, self.type3 = {}, {}, {}    
    classes = set(self.ontology.classes())    
    self.classes_dict = {onto_class: onto_class.lex for onto_class in classes}
    self.classesIntoTypes(classes)    
    self.majority_count = 0  

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
    
  def getMajorityClass(self, polarity):
    index = (pd.get_dummies(pd.DataFrame(np.concatenate([np.array(['negative','neutral','positive']), polarity])))
            .values[3:,:].sum(0).argmax())
    if index == 0: return np.array([[1, 0, 0]])
    elif index == 1: return np.array([[0, 1, 0]])
    else: return np.array([[0, 0, 1]])    

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
      prediction = self.majority_class
      self.majority_count += 1
    return prediction      

  def run(self, train_data, test_data, purpose):
    self.majority_class = self.getMajorityClass(train_data[:, 2])
    
    predictions = np.zeros([1,3])
    for sentence, aspect, _ in test_data:
      predictions = np.concatenate([predictions, self.predictSentiment(sentence, aspect)])
    predictions = predictions[1:,:]
    
    self.evaluation(test_data[:, 2], predictions, purpose)
    print('')
    return predictions  
   
  def evaluation(self, Y, pred, purpose, print_results=True):
    real = (pd.get_dummies(pd.DataFrame(np.concatenate([np.array(['negative','neutral','positive']), Y]))).values[3:,:])
    pos_pos,pos_neu,pos_neg,neg_pos,neg_neu,neg_neg=0,0,0,0,0,0
    for i in range(len(pred)):
      if (pred[i].argmax() == 0) and (real[i].argmax()==0): neg_neg += 1
      if (pred[i].argmax() == 0) and (real[i].argmax()==1): neg_neu += 1
      if (pred[i].argmax() == 0) and (real[i].argmax()==2): neg_pos += 1
            
      if (pred[i].argmax() == 2) and (real[i].argmax()==0): pos_neg += 1
      if (pred[i].argmax() == 2) and (real[i].argmax()==1): pos_neu += 1
      if (pred[i].argmax() == 2) and (real[i].argmax()==2): pos_pos += 1
                
    pos_pred = pos_pos + pos_neu + pos_neg
    neg_pred = neg_pos + neg_neu + neg_neg    
        
    pos_true = pos_pos + neg_pos
    neu_true = pos_neu + neg_neu
    neg_true = pos_neg + neg_neg    
    total = pos_true + neu_true + neg_true
        
    table = pd.DataFrame(columns = ['Negative', 'Neutral',  'Positive',  '|', 'total'],
                         data =   [[ neg_neg,    neg_neu,     neg_pos,   '|', neg_pred], 
                                   [ pos_neg,    pos_neu,     pos_pos,   '|', pos_pred],
                                   ['--------', '-------',   '--------', '-', '-----'],
                                   [ neg_true,   neu_true,    pos_true,  '|',  total]],
                         index =   ['Negative', 'Positive', '--------', 'total'])
    
    acc = np.round(100*sklearn.metrics.accuracy_score(np.argmax(pred, 1), np.argmax(real, 1)), 2)
    loss = sklearn.metrics.log_loss(real, pred)      
    path = './Results/logs/{0}/{1}/'.format(self.__class__.__name__, purpose)
    results = {'accuracy' : acc, 'loss' : loss, 'table' : table.to_json()}
    if not os.path.exists(path):
      os.makedirs(path)
    with open(path+'results.json', 'w') as file:
      file.write(json.dumps(results)) 
    if print_results:
      print("                  "+purpose+" Data")
      print(table)
      print('Accuracy:', acc)
      print('Loss:', loss)


if __name__ == '__main__':
  np.set_printoptions(threshold=np.inf, edgeitems=3, linewidth=120)
  
  FLAGS = tf.app.flags.FLAGS
  for name in list(FLAGS):
    if name not in ('showprefixforinfo',):
      delattr(FLAGS, name)

  tf.app.flags.DEFINE_string('f', '', 'kernel')
  tf.app.flags.DEFINE_float('train_proportion', 1.0, 'Train proportion for train/validation split')
  tf.app.flags.DEFINE_integer('seed', None, 'Random seed')
  tf.app.flags.DEFINE_string('ontology_path', './Ontology/Ontology_restaurants.owl', 'Ontology path')
  tf.app.flags.DEFINE_boolean('train_model', True, 'Run Ont on train data')
  tf.app.flags.DEFINE_boolean('predict_values', True, 'Run Ont on test data')  
  tf.app.flags.DEFINE_string('train_data_path', './Data/ABSA-15_Restaurants_Train_Final.xml', 'Train data path')
  tf.app.flags.DEFINE_string('test_data_path', './Data/ABSA15_Restaurants_Test.xml', 'Test data path')

  reader = Reader(FLAGS)

  data_train, _ = reader.readData(FLAGS.train_data_path)
  data_test, _ = reader.readData(FLAGS.test_data_path)

  model = Ont(FLAGS.ontology_path)

  if FLAGS.train_model:
    print('Training...')
    a = datetime.datetime.now()
    predictions_train = model.run(data_train, data_train, 'Train')
    b = datetime.datetime.now()
    print('Time:', b-a)

  if FLAGS.predict_values:
    print('Prediction...')
    c = datetime.datetime.now()
    predictions_test = model.run(data_train, data_test, 'Test')  
    d = datetime.datetime.now()
    print('Time:', d-c)