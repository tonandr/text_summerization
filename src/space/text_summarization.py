'''
Created on Apr 26, 2019

@author: Inwoo Chung (gutomitai@gmail.com)
'''
    
import os
import glob
import argparse
import time
import struct, codecs

import pandas as pd
import numpy as np
import scipy.io as io

from keras.preprocessing.text import text, Tokenizer
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, GRU, Dropout
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.utils.data_utils import Sequence

import ipyparallel as ipp

# Constant.
IS_DEBUG = False

IS_MULTI_GPU = False
NUM_GPUS = 4
MAX_NUM_WORDS = 200000
EMBEDDING_DIM = 100
ARTICLE_MAX_SEQUENCE_LENGTH = 781
SUMMARY_MAX_SEQUENCE_LENGTH = 56

class TextSummarizer(object):
    """Text summarizer."""
    MODEL_FILE_NAME = 'text_summarizer.hd5'
    
    def __init__(self, raw_data_path, glove_file_name, hps, model_loading):        
        # Initialize.
        self.raw_data_path = raw_data_path
        self.hps = hps
        self.model_loading
        
        # Assign symbol indexing objects.
        self.embedding = {}
        
        with open(os.path.join(raw_data_path, glove_file_name), 'r', encoding='utf-8') as f:
            for v in f:
                vals = v.split() #?
                word = vals[0]
                coeffs = np.asfarray(vals[1:])
                self.embedding[word] = coeffs 
        
        vocab_df = pd.read_csv(os.path.join(raw_data_path, 'vocab'), sep=' ', header=None)
        texts = list(vocab_df.iloc[:, 0])
        self.tokenizer = Tokenizer(filters='', lower=True, split=' ', char_level=False, oov_token='oov')
        self.tokenizer.fit_on_texts(texts)     
        self.tokenizer.word_index = {e:i for e,i in self.tokenizer.word_index.items() \
                                     if i <= MAX_NUM_WORDS}
        self.tokenizer.index_word = {i:e for e,i in self.tokenizer.word_index.items() \
                                     if i <= MAX_NUM_WORDS}
         
        self.embedding_matix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM)) #?
                    
        # Create the glove pre_trained embedding matrix.
        for word, i in self.tokenizer.word_index.items():
            
            try:
                embedding_vec = self.embedding[word]
            except KeyError:
                continue
            
            self.embedding_matix[i] = embedding_vec        
                
        # Create the embedding layer.        
        self.embedding_layer_article = Embedding(MAX_NUM_WORDS
                               , EMBEDDING_DIM
                               , weights = [self.embedding_matrix]
                               , input_length = ARTICLE_MAX_SEQUENCE_LENGTH
                               , trainable = False)
        self.embedding_layer_summary = Embedding(MAX_NUM_WORDS
                               , EMBEDDING_DIM
                               , weights = [self.embedding_matrix]
                               , input_length = SUMMARY_MAX_SEQUENCE_LENGTH
                               , trainable = False)
        self.embedding_layer_summary_gen = Embedding(MAX_NUM_WORDS
                               , EMBEDDING_DIM
                               , weights = [self.embedding_matrix]
                               , input_length = 1
                               , trainable = False)
        
        if model_loading == True:
            print('Load the pre-trained model...')

            if IS_MULTI_GPU == True:
                self.model = multi_gpu_model(load_model(self.MODEL_FILE_NAME), gpus = NUM_GPUS) 
            else:
                self.model = load_model(self.MODEL_FILE_NAME)
        else:
            # Design the model.
            print('Design the model.')
            
            # Input1: article text, n (n sequence), (token index).   
            input1 = Input(shape=(ARTICLE_MAX_SEQUENCE_LENGTH,))
            x1 = self.embedding_layer_article(input1)
            
            _, c = GRU(self.hps['gru1_dim'], return_state = True, name='gru1')(x1)
            
            # Input2: summary text, n (n sequence), (token index). 
            input2 = Input(shape=(SUMMARY_MAX_SEQUENCE_LENGTH,))
            x2 = self.embedding_layer_summary(input2)
            
            x, _ = GRU(self.hps['gru2_dim']
                     , return_sequences = True
                     , return_state = True
                     , name='gru2')(x2, initial_state = c)
            
            output = Dense(MAX_NUM_WORDS
                                   , activation = 'softmax'
                                   , name = 'dense1')(x)
                                            
            # Create the model.
            if IS_MULTI_GPU == True:
                self.model = multi_gpu_model(Model(inputs=[input1, input2]
                                                   , outputs=[output]), gpus = NUM_GPUS)
            else:
                self.model = Model(inputs=[input1, input2], outputs=[output])  
        
            # Compile the model.
            optimizer = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')
            self.model.summary()

        self._make_summary_gen_models()
            
    def _make_summary_gen_models(self):
        """Make summary generation model."""
        input1 = Input(shape=(ARTICLE_MAX_SEQUENCE_LENGTH,))
        _, c = self.model.get_layer('gru1')(input1)
                
        self.article_f_gen = Model([input1], [c])

        input2 = Input(shape=(1, ))
        x = self.embedding_layer_summary_gen(input2)

        r = Input(shape=(self.hps['gru2_dim'],))
        x, c = self.model.get_layer('gru2')(x, initial_state = r)
        output = self.model.get_layer('dense1')(x)         
    
        self.summary_gen_model = Model([input1, r], [output, c])

    class TrainingSequence(Sequence):
        """Training data set sequence."""
        
        def __init__(self, raw_data_path, hps, tokenizer):
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.tokenizer = tokenizer
            self.samples = []
                
            with open(os.path.join(raw_data_path, 'train.bin'), 'rb') as f:
                while True:
                    len_bytes = f.read(8)
                    if not len_bytes: break # finished reading this file
                    str_len = struct.unpack('q', len_bytes)[0]
                    sample = struct.unpack('%ds' % str_len, f.read(str_len))[0]
                    self.samples.append(sample)
            
            self.batch_size = len(self.samples) // hps['step_per_epoch']
                
        def __len__(self):
            return self.hps['step_per_epoch']
        
        def __getitem__(self, index):
            # Check the last index.
            # TODO
            
            articles = []
            summary_paras_b = []
            summary_paras_t = []
            
            for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                sample = str(self.samples[bi])
                sample_s = sample.split('</s>')
                
                # Check exception.
                if len(sample_s) == 0: continue #?
                
                # Summary.
                summary = sample_s[0].split('<s>')[-1] + ' '
                
                for i in range(1, len(sample_s) - 1):
                    summary += sample_s[i].split('<s>')[1] + ' ' #?
                
                summary = np.asarray(self.tokenizer.texts_to_sequences([summary])[0])
                
                if len(summary) >= SUMMARY_MAX_SEQUENCE_LENGTH + 1:
                    summary_b = summary[:SUMMARY_MAX_SEQUENCE_LENGTH] #?
                    summary_t = summary[1:(SUMMARY_MAX_SEQUENCE_LENGTH + 1)]
                elif len(summary) == SUMMARY_MAX_SEQUENCE_LENGTH:
                    summary_b = summary[:SUMMARY_MAX_SEQUENCE_LENGTH]
                    summary_t = np.concatenate([summary[1:SUMMARY_MAX_SEQUENCE_LENGTH], summary[-1]])
                else:
                    redun = np.zeros(shape=(SUMMARY_MAX_SEQUENCE_LENGTH - len(summary))) #?
                    redun[:] = summary[-1]
                    summary_b = np.concatenate([summary, redun])
                    summary_t = np.concatenate([summary_b[1:SUMMARY_MAX_SEQUENCE_LENGTH], summary[-1]])
                
                summary_paras_b.append(summary_b)
                summary_paras_t.append(summary_t)
                
                # Article.
                sample_f = sample_s[-1]
                sample_f = sample_f.replace('. \'\'', '.')
                sample_f = sample_f.replace('? \'\'', '?')
                sample_f = sample_f.replace('! \'\'', '!')
                
                sample_f_s = sample_f.split('.')
                
                article = ''
                for s in sample_f_s:
                    article += s + ' '                
 
                article = np.asarray(self.tokenizer.texts_to_sequences([article])[0])
                
                if len(article) > ARTICLE_MAX_SEQUENCE_LENGTH:
                    article = article[:ARTICLE_MAX_SEQUENCE_LENGTH]
                elif len(article) < ARTICLE_MAX_SEQUENCE_LENGTH:
                    redun = np.zeros(shape=(ARTICLE_MAX_SEQUENCE_LENGTH - len(article))) #?
                    redun[:] = article[-1]
                    article = np.concatenate([article, redun])                    
                
                articles.append(article)
                
            articles = np.asarray(articles)
            summary_paras_b = np.asarray(summary_paras_b)
            summary_paras_t = np.asarray(summary_paras_t)
            
            return ({'input1': articles, 'input2': summary_paras_b}, {'output': summary_paras_t})
        
    def train(self):
        """Train."""        
        # Get training data generator.
        trGen = self.TrainingSequence(self.raw_data_path, self.hps)
        hist = self.model.fit_generator(trGen
                      , steps_per_epoch=self.hps['step_per_epoch']                  
                      , epochs=self.hps['epochs']
                      , verbose=1
                      , max_queue_size=100
                      , workers=4
                      , use_multiprocessing=True)
                        
        # Print loss.
        print(hist.history['loss'][-1])
        
        print('Save the model.')            
        self.model.save(self.MODEL_FILE_NAME)
        
        # Calculate loss.
        lossList = list()
        lossList.append(hist.history['loss'][-1])
            
        lossArray = np.asarray(lossList)        
        lossMean = lossArray.mean()
        
        print('Each mean loss: {0:f} \n'.format(lossMean))
        
        with open('losses.csv', 'a') as f:
            f.write('{0:f} \n'.format(lossMean))
                
        with open('loss.csv', 'w') as f:
            f.write(str(lossMean) + '\n') #?
            
        return lossMean 

    def test(self, article_file_path, output_path):
        """Test."""
        with open(article_file_path, 'r') as f:
            val_texts = f.readlines()
            
        with open(output_path, 'w') as f:
            for text in val_texts:
                # Make a tokenized sequence.
                sample_f = text
                sample_f = sample_f.replace('. \'\'', '.')
                sample_f = sample_f.replace('? \'\'', '?')
                sample_f = sample_f.replace('! \'\'', '!')
                
                sample_f_s = sample_f.split('.')
                
                article = ''
                for s in sample_f_s:
                    article += '<s> ' + s + ' </s> '                
 
                article = np.asarray(self.tokenizer.texts_to_sequences([article])[0])
                
                if len(article) > ARTICLE_MAX_SEQUENCE_LENGTH:
                    article = article[:ARTICLE_MAX_SEQUENCE_LENGTH]
                elif len(article) < ARTICLE_MAX_SEQUENCE_LENGTH:
                    redun = np.zeros(shape=(ARTICLE_MAX_SEQUENCE_LENGTH - len(article))) #?
                    redun[:] = article[-1]
                    article = np.concatenate([article, redun])                 
                
                # Get summary.
                article = article[np.newaxis, ...]
                _, c = self.article_f_gen.predict(article)
                
                summary = []
                s = np.asarray([[self.tokenizer.word_index['<s>']]])
                summary.append(s)
                
                for _ in range(SUMMARY_MAX_SEQUENCE_LENGTH - 1):
                    s_p, c = self.summary_gen_model.predict([s, c]) # One hot vector.
                    s_p = s_p.ravel()
                    
                    while(s_p.sum() > 1.0):
                        s_p = s_p / s_p.sum()
            
                    s = np.random.choice(s_p.shape[0]
                                                   , 1
                                                   , p = s_p)[0] 
                    
                    summary.append(s)
                
                summary = self.tokenizer.sequences_to_texts([summary])[0] #?                    
                summary_s = summary.split('.') #?: ?, !

                for v in summary_s: # Number limitation?
                    f.write('<s> ' + v + '. </s> ')
                
                f.write('\n')

def main(args):
    """Main.
    
    Parameters
    ----------
    args : argument type 
        Arguments
    """ 
    hps = {}

    if args.mode == 'train':
        # Get arguments.
        raw_data_path = args.raw_data_path
        glove_file_name = args.glove_file_name
        
        # hps.
        hps['gru1_dim'] = int(args.gru1_dim)
        hps['gru2_dim'] = int(args.gru2_dim)  
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay)
        hps['step_per_epoch'] = int(args.step_per_epoch)
        hps['epochs'] = int(args.epochs) 
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Train.
        text_summarizer = TextSummarizer(raw_data_path, glove_file_name, hps, model_loading)
        
        ts = time.time()
        text_summarizer.train()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif args.mode == 'test':
        # Get arguments.
        raw_data_path = args.raw_data_path
        glove_file_name = args.glove_file_name
        article_file_path = args.article_file_path
        output_path = args.output_path
      
        # hps.
        hps['gru1_dim'] = int(args.gru1_dim)
        hps['gru2_dim'] = int(args.gru2_dim)  
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay)
        hps['step_per_epoch'] = int(args.step_per_epoch)
        hps['epochs'] = int(args.epochs) 
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Train.
        text_summarizer = TextSummarizer(raw_data_path, glove_file_name, hps, model_loading)
        
        ts = time.time()
        text_summarizer.test(article_file_path, output_path)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
        
if __name__ == '__main__':
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode')
    parser.add_argument('--raw_data_path')
    parser.add_argument('--glove_file_name')
    parser.add_argument('--article_file_path')    
    parser.add_argument('--output_path')
    parser.add_argument('--gru1_dim')
    parser.add_argument('--gru2_dim')
    parser.add_argument('--lr')
    parser.add_argument('--beta_1')
    parser.add_argument('--beta_2')
    parser.add_argument('--decay')
    parser.add_argument('--step_per_epoch')
    parser.add_argument('--epochs')
    parser.add_argument('--model_loading')
    args = parser.parse_args()
    
    main(args)      