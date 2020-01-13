import xml.etree.ElementTree as ET
import re
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import itertools

class Reader(): 
  class Review():
    def __init__(self):
      self.id = None
      self.sentences = {}
    def __str__(self):
      text = 'REVIEW: {}\n'.format(self.id)
      for sentence in self.sentences.values():
        text += str(sentence)
      return text           
  class Sentence():
    def __init__(self):
      self.id = None
      self.text = ''
      self.opinions = []
    def __str__(self):
      text = 'SENTENCE: {}\n{}\n'.format(self.id, self.text)
      if self.opinions:
        text += 'SENTENCE-LEVEL OPINIONS:\n'
        for opinion in self.opinions:
          text += str(opinion) + "\n"
      return text   
  class Opinion():
    def __init__(self):
      self.entity, self.attribute = None, None
      self.category, self.polarity, self.target = None, None, None
      self.start, self.end = None, None
    def __str__(self):
      if self.target:
        text = "[{}; {}] '{}' ({}-{})".format(self.category, self.polarity, self.target, self.start, self.end)
      return text

  def __init__(self, FLAGS):
    self.train_proportion = FLAGS.train_proportion
    self.seed = FLAGS.seed
    
  def cleaning(self, text):
    text = re.sub("(&quot;)", '"', text)
    text = re.sub("(&apos;)", "'", text)
    text = re.sub("(&amp;)", "and", text) 
    text = re.sub(r"[^\w\s]", " ", text)  # Replace punctuation with spaces
    text = re.sub("\d", "", text)         # Remove numbers
    text = re.sub("\s+", " ", text)       # Replace all spaces with 1 space
    text = re.sub("^\s+|\s+$", "", text)  # Remove spaces in the beginning and in the end
    return text    
  
  def readData(self, file):
    self.reviews = {}
    tree = ET.parse(file)
    root = tree.getroot()
    sentences, aspects, polarities = [],[],[]
    longest_sentence = 0
    for R in root:
      review = self.Review()
      for s in R.find('sentences').findall('sentence'):
        sentence = self.Sentence()
        sentence.text = self.cleaning(s.find('text').text.lower())
        if s.get('OutOfScope'):
          continue
        else:
          if s.find('Opinions'):
            for o in s.find('Opinions').findall('Opinion'):
              opinion = self.Opinion()
              opinion.target = self.cleaning(o.get("target").lower())
              if ((len(opinion.target) > len(sentence.text)) or
                  (sentence.text.find(opinion.target) == -1) or
                  (opinion.target == "null")):
                   continue            
              if opinion.target != '':
                opinion.category = o.get('category').lower()
                opinion.entity, opinion.attribute = opinion.category.split('#')
                opinion.polarity = o.get('polarity').lower()
                opinion.start = int(o.get('from'))
                opinion.end = int(o.get('to'))
                sentence.opinions.append(opinion)
                sentences.append(sentence.text)
                aspects.append(opinion.target)
                polarities.append(opinion.polarity)  
            sentence.id = s.get('id')        
            review.sentences[sentence.id] = sentence
            if longest_sentence < len(sentence.text.split()):
              longest_sentence = len(sentence.text.split())
      review.id = R.get('rid')
      self.reviews[review.id] = review
    sentences = np.array(sentences).reshape([-1,1])
    aspects = np.array(aspects).reshape([-1,1])
    polarities = np.array(polarities).reshape([-1,1])
    data = np.concatenate([np.concatenate([sentences, aspects], axis=1), polarities], axis=1)
    return data, longest_sentence
  
  def splitTrainData(self, data):
    X_train, X_test, y_train, y_test = train_test_split(data[:,:2], data[:,2], 
                                                        train_size=self.train_proportion, random_state=self.seed)
    
    data_train = np.concatenate([X_train, y_train.reshape([-1,1])], axis=1)
    data_test = np.concatenate([X_test, y_test.reshape([-1,1])], axis=1)
    return data_train, data_test  
    
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
      if polarity == 'negative': polarities.append(np.array([1,0,0]))
      elif polarity == 'neutral': polarities.append(np.array([0,1,0]))
      elif polarity == 'positive': polarities.append(np.array([0,0,1]))
    sentences = np.array(list(itertools.zip_longest(*sentences, fillvalue=0))).T
    aspects = np.array(list(itertools.zip_longest(*aspects, fillvalue=0))).T
    polarities = np.array(polarities)
    data = [sentences, aspects, polarities]
    return data 

  def readEmbeddings(self, file):
    word2idx = {'PAD': 0 }
    embeddings = []    
    with open(file, 'r', encoding='utf8') as f: 
      for index, line in enumerate(f): 
        values = line.split()
        word = values[0]
        word_embeddings = np.asarray(values[1:], dtype=np.float32)
        word2idx[word] = index + 1
        embeddings.append(word_embeddings) 
    EMBEDDINGS_DIM = len(embeddings[0])
    embeddings.insert(0, np.zeros(EMBEDDINGS_DIM))
    embeddings.append(np.random.randn(EMBEDDINGS_DIM))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    word2idx['UNK'] = len(embeddings)
    idx2word = {v: k for k, v in word2idx.items()}
    return embeddings, word2idx, idx2word 

  def aspectCategories(self, purpose):
    categories = []
    for r in self.reviews.values():
      for s in r.sentences.values():
        for o in s.opinions:
          categories.append(o.category)
    categories = pd.DataFrame(categories, columns=['category'])      
    cat_order = pd.unique(categories.category)
    cat_order.sort()
    g = sns.catplot('category', data=categories, kind='count', aspect=1.5, 
                    palette=sns.color_palette('gray', len(cat_order)),
                    order = cat_order)
    g.set_xticklabels(rotation=90, size=13)
    g.set_xlabels(size=15)
    g.set_ylabels(size=15)
    plt.title(purpose.title()+ ' data', fontsize=20, y=1.05)
    if not os.path.exists('./Results'):
      os.makedirs('./Results')
    plt.savefig('./Results/{}_aspects.png'.format(purpose), bbox_inches='tight')
    plt.show()
    
  def activationFunctions(self):
    sigmoid = lambda x: 1/(1+np.exp(-x))
    x = np.arange(-5,5,0.1)
    y= np.empty([2,100])
    y[0,:]= np.tanh(x)
    y[1,:] = sigmoid(x)
    i = 0
    for fun in ['tanh', 'sigmoid']:
        fig = plt.figure(fun)
        ax = fig.add_subplot(111)
        ax.plot(x,y[i,:])
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')    
        ax.axhline(linewidth = 0.5, color = 'black')
        ax.axvline(linewidth = 0.5, color = 'black')
        plt.title('y = '+fun+'(x)')
        plt.show()
        i += 1    

  def transformPolarity(self, polarity):
    if polarity == 'negative': return np.array([1,0,0])
    elif polarity == 'neutral': return np.array([0,1,0])
    elif polarity == 'positive': return np.array([0,0,1])
    else: Exception(polarity) 

  def plotAttention(self, weights, sentence, file):
    plt.figure('attention', figsize=(14,14))
    weights = weights[weights!=0]
    labels = np.array([f'{sentence}\n{weights:.2f}' for sentence, weights in zip(sentence, weights)]).reshape([1,-1])
    sns.heatmap([weights], annot=labels, yticklabels=False, xticklabels=False, cbar=False, square=True, cmap='binary', fmt='', annot_kws={'size':15})
    plt.savefig(file)

  def plotPolarityDistribution(self, data_train, data_test, print_results, plot_results):
    train = np.array(list(map(self.transformPolarity, np.array(data_train)[:,2])))
    test = np.array(list(map(self.transformPolarity, np.array(data_test)[:,2])))
    total_train = train.sum()
    total_test = test.sum()
    
    nega, neua, pa = train.sum(0)
    max_height = np.max([nega,neua,pa])
    negb, neub, pb = test.sum(0)

    neg_train = np.ones([nega])*1.6
    neg_test = np.ones([negb])*1.8
    neu_train = np.ones([neua])*0.9
    neu_test = np.ones([neub])*1.1
    pos_train = np.ones([pa])*0.2
    pos_test = np.ones([pb])*0.4    
    
    if print_results:
        table = pd.DataFrame(columns = pd.MultiIndex.from_product([['Negative', 'Neutral', 'Positive', 'total'], 
                                                                  ["Freq.", "%"]]),
                            data    =  [[nega, np.round(100*nega/total_train,2),
                                         neua, np.round(100*neua/total_train,2), 
                                         pa, np.round(100*pa/total_train,2), 
                                         total_train, np.round(100*total_train/total_train)],
                                        [negb, np.round(100*negb/total_test,2), 
                                         neub, np.round(100*neub/total_test,2),
                                         pb, np.round(100*pb/total_test,2),
                                         total_test, np.round(100*total_test/total_test)]],
                            index   =  ["Train", "Test"])
        print(table)
        
    if plot_results:
      fig = plt.figure("Polarity distribution", figsize=(8, 3))
      fig.set_size_inches(8, 3.3, forward=True)
      ax = fig.add_subplot(111)
      
      ax.spines["left"].set_position("zero")
      ax.spines["right"].set_color("none")
      ax.spines["top"].set_color("none")
      ax.spines["left"].set_smart_bounds(True)
      ax.spines["bottom"].set_smart_bounds(True)
      ax.xaxis.set_ticks([])
      ax.yaxis.set_ticks_position("left")
      ax.yaxis.set_label_text("Number of sentences")
      ax.set_ylim([0, max_height+100])
      
      green = sns.color_palette("dark", 2)[1]
      red = sns.color_palette("OrRd",10)[8]
      ax.hist(pos_train, label = "Train positive", color=green, hatch = "//")
      ax.hist(pos_test, label = "Test positive", color=green, hatch = "\\\\")
      ax.hist(neu_train, label = "Train neutral", color="grey", hatch = "//")
      ax.hist(neu_test, label = "Test neutral", color="grey", hatch = "\\\\")
      ax.hist(neg_train, label = "Train negative", color=red, hatch = "//")
      ax.hist(neg_test, label = "Test negative", color=red, hatch = "\\\\")
      
      ax.annotate("{}%".format(np.round(100*pa/total_train,2)), xy=(0.12, 1325))
      ax.annotate("{}%".format(np.round(100*pb/total_test,2)), xy=(0.35, 495))
      ax.annotate("{}%".format(np.round(100*neua/total_train,2)), xy=(0.85, 85))
      ax.annotate("{}%".format(np.round(100*neub/total_test,2)), xy=(1.05, 40))  
      ax.annotate("{}%".format(np.round(100*nega/total_train,2)), xy=(1.52, 502))
      ax.annotate("{}%".format(np.round(100*negb/total_test,2)), xy=(1.73, 150))
      plt.legend(loc="upper right")
      plt.show()