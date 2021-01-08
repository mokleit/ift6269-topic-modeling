# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:22:52 2020

@author: rober
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:02:42 2020

@author: rober
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import time
from scipy import special
import math
import os 
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
#import resource
import os
import re

debug=False  

folder = "C:/Users/rober/OneDrive/Bureau/etude/graph models udem/Projet/ift6269-topic-modeling/lda-python/data/scribenotes"

n_features = 1000 #build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
n_top_words = 20 #for prints
max_treshold = 0.9 # high frequency words
min_treshold = 2 #low frequency words



input_file = folder +"/corpus.txt"


    
documents = []

with open(input_file) as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split( r'\n')]
        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        documents.append(inner_list[0])

#Prepare data
        
stops = [
"a", "about", "above", "across", "after", "afterwards", "again", "against",
"all", "almost", "alone", "along", "already", "also", "although", "always",
"am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
"any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
"around", "as", "at", "back", "be", "became", "because", "become",
"becomes", "becoming", "been", "before", "beforehand", "behind", "being",
"below", "beside", "besides", "between", "beyond", "bill", "both",
"bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
"could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
"down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
"elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
"everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
"find", "fire", "first", "five", "for", "former", "formerly", "forty",
"found", "four", "from", "front", "full", "further", "get", "give", "go",
"had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
"hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
"how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
"interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
"latterly", "least", "less", "ltd", "made", "many", "may", "me",
"meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
"move", "much", "must", "my", "myself", "name", "namely", "neither",
"never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
"nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
"once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
"ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
"please", "put", "rather", "re", "same", "see", "seem", "seemed",
"seeming", "seems", "serious", "several", "she", "should", "show", "side",
"since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
"something", "sometime", "sometimes", "somewhere", "still", "such",
"system", "take", "ten", "than", "that", "the", "their", "them",
"themselves", "then", "thence", "there", "thereafter", "thereby",
"therefore", "therein", "thereupon", "these", "they", "thick", "thin",
"third", "this", "those", "though", "three", "through", "throughout",
"thru", "thus", "to", "together", "too", "top", "toward", "towards",
"twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
"very", "via", "was", "we", "well", "were", "what", "whatever", "when",
"whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
"who", "whoever", "whole", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself",
"yourselves""probabilistic" ,"graphical" ,"models" ,"fall" ,"lecture",
 "september", "lecturer", "simon", "lacoste" ,"julien", "scribe",
'william' ,'chelle', "isabela", "albuquerque",  "disclaimer" ,"these" ,
"notes", "have", "only", "been" ,"lightly" ,"proofread",
"philippe" ,"brouillard", "tristan" ,"deleu", "bastien", "lachapelle" ,
  "zakaria", "soliman", "eeshan", "gunesh", "dhekane", "younes","driouiche" ,
  "martin", "weiss" , "beardsell", "jaime", "roquero" ,"jieying","proofread" , 
  "pravish" ,"sainath", "ismael", "martinez" ,"abdelrahman", "zayed",
"able","access","according","achieve","acts","actually",
"advantage","advantages","algorithm","algorithms",
"argmax","argument","assume","assumed","assuming",
"assumptions","average","averaging","away","zero",
"word","words","works","world","worry","write","writing","yield",
"yields","want","ways","true","trying","turn","turns","types","zero",
"october","december","cours","november","definition","variable" ,"variables" 
"random" ,"called","gives",'consider','example','know' ,'note' ,'random',
'function' , 'means','given' ,'model','variables','data']



cv = CountVectorizer(max_df=max_treshold, min_df=min_treshold,
                       max_features=n_features,
                       stop_words=stops)



DOC = cv.fit_transform(documents)

Vocabulary = [str.strip('_') for str in cv.get_feature_names()]
#get a matrix of counts for each element of vocabulary
Vocab_per_Doc_count = DOC.toarray()

# Number of Documents M/ Size of Vocabulary V 
M , V = DOC.shape
K = 8

print(len(Vocabulary))

import gc
gc.collect()
''' 
Doc_words :list of words for each document d (words are represented by their index in Vocabulary) [M x N[d]]
N :list of size of each document (number of words)
'''
def DataTrans(x):
    """Turn the data into the desired structure"""
    
    N_d = np.count_nonzero(x)
    V = len(x)
    
    row = 0
    
    doc = np.zeros((N_d, V))
    for i in range(V):
        if x[i] == 0:
            pass
        else:
            doc[row, i] = x[i]
            row += 1
    
    return doc


docs = list(map(DataTrans, Vocab_per_Doc_count))


def log_gamma( x):
        return  special.loggamma(x)
    
def psi( x):
    '''
    Called di_gamma in the Blei-C  implementation
    
    '''
    return special.polygamma(0, x)


def psi_2( x):
    '''
    Called tri_gamma in the Blei-C  implementation
    
    '''
    return special.polygamma(1, x)


def gradient(alpha,gamma ):
    '''
    Called d_alhood in the Blei-C  implementation
    see annexe A.4.2 for derivation
    '''
    M = gamma.shape[0]

    ss = sufficient_statistic(gamma)
    
    D_alpha=  M * (psi(alpha.sum())-psi(alpha)) + ss
    return D_alpha

def sufficient_statistic(x):
    '''
    COmpute the sufficient statistic from the gamma matrix
    '''
    ss= (psi(x)-psi(x.sum(axis=1,keepdims=True))).sum(axis=0)
    return ss

def update_alpha_hessian_trick(alpha,gamma):
    '''
    newton update 
    see annexe A.2 for derivation
    '''
    N_doc= gamma.shape[0]
    
    D1_alpha = gradient(alpha,gamma ) 
    
    D2_alpha_diagonal =  - psi_2(alpha)*N_doc
    
    z=  psi_2(alpha.sum())*N_doc

    c= (D1_alpha/D2_alpha_diagonal).sum() / (1/z +  (1/D2_alpha_diagonal).sum())

    update =  (D1_alpha -c)/D2_alpha_diagonal
    return alpha - update

def newton_alpha(alpha,gamma):
    ''' 
    run newton update until convergence of parameter
    input: 
    gamma matrix from the m step
        
    '''
    I = 0
    converged = False
    optimal_alpha= alpha
    while not converged:
      
      optimal_alpha_old = optimal_alpha
      optimal_alpha = update_alpha_hessian_trick(optimal_alpha ,gamma)
      delta = np.linalg.norm(optimal_alpha- optimal_alpha_old) 
      
      if (delta < 10e-3 or I >100):
          converged = True   
          print('stoped after:',I,'iterations')
      I += 1
      
    return optimal_alpha

#==========================================================================================================
#==========================================================================================================

def e_step(beta_t ,alpha_t, docs,K):
  '''
  Input:
      beta_t: [K x V] matrix (beta_i,j = p(w^j=1|z^i=1))
      alpha_t: [1 x K] vector 
      docs: list of of documents and their words: [M X N[d] X V]
      
      an iteration computation of optimal phi and gamma
      phi   : variational multinomial_parameters [M x N[d] x K]
      gamma : variational dirichlet parameter [M x K]
      N  : list of number of words in each document


  '''
  M =len(docs)
  N=[doc.shape[0] for doc in docs]
  #Initialization
  optimal_phi = []
  optimal_gamma = np.zeros((M,K))
  
  #iterate for each document
  for d in range(M):
    doc= docs[d]
    #initialization for  each document
    optimal_phi_doc = 1/K * np.ones((N[d],K))#(1)
    optimal_gamma[d] = alpha_t + np.max((N[d]/K,0.2))  #(2) added a minimum value so that the psi doesnt create overflow
    converged = False

    while not converged:

      old_optimal_gamma = optimal_gamma[d]
      old_optimal_phi_doc = optimal_phi_doc

      # update phi
      optimal_phi_doc = (doc@beta_t.T) * np.exp(psi(optimal_gamma[d])-psi(optimal_gamma[d].sum())) #(6)
      optimal_phi_doc = optimal_phi_doc / (np.sum(optimal_phi_doc,axis=1)[:,None]) #(7)
      # update gamma
      optimal_gamma[d] = alpha_t + np.sum(optimal_phi_doc,axis = 0) # (8)
      # check convergence
      if (np.linalg.norm(optimal_gamma[d] - old_optimal_gamma) < 10e-3 and np.linalg.norm(optimal_phi_doc - old_optimal_phi_doc) < 10e-3):
        converged = True
    optimal_phi.append(optimal_phi_doc)
    
  return optimal_phi,optimal_gamma

def m_step(phi_t ,gamma_t ,initial_alpha, V ,docs):
  '''
    
  inputs:
     phi_t : phi paramters from the E-step, shape = [M x N[d] x K] (vary per document)
     gamma_t: matrix of gamma parameter from the E-step,  shape  = 1 x K
     initial_alpha: matrix of gamma parameter from the E-step,  shape  =N document X N topics
     V: n_features
     words: list of  list of word index present in each of the document ,shape = M document X N[d] words  (vary per document)
     
  an iteration computation of optimal beta and alpha
     Output:
         
  beta   : [K x V]
  alpha  : dirichlet parameter [K]
  
  '''
  #initialization
  M, K = gamma_t.shape
  optimal_beta = np.zeros((K,V))
  optimal_alpha =  np.zeros((1,K))

  # update beta
  beta_per_doc = np.zeros((M,K,V))
  for d in range(M):

    beta_per_doc[d] =   phi_t[d].T @ docs[d]
  
  optimal_beta = np.sum(beta_per_doc, axis=0)
  #Normalization of beta  
  optimal_beta = optimal_beta / (np.sum(optimal_beta, axis= 1))[:,np.newaxis]

  if debug:
    print("The next array should contain only ones")
    print(np.sum(optimal_beta,axis = 1))
    print(optimal_alpha.shape)
    
  # update alpha : we use Newton Raphson method to update alpha
  optimal_alpha = newton_alpha(initial_alpha,gamma_t)
      
  return optimal_beta,optimal_alpha

#=============================================================================================================
#=============================================================================================================
  

  
def run_em(docs,K,max_iter,V):
    '''
    run the E-step and M-Step iteratively until the log likelihood converges
    returns the final parameters
    
    inputs:
        N: number of words in each document 
        Doc_words: list of  list of word index present in each of the document ,shape = M document X N words  
        Initial_alpha: initial alpha parameters used
        V: n_features
    '''

    #initialisation
    optimal_beta = np.random.dirichlet(np.ones(V),K) #1/V * np.ones((K,V))
    initial_alpha =np.random.gamma(shape= np.ones((K)), scale = 0.01)
    converged = False
    I = 0

    #Run EM Algorithm
    while not converged:

         #E-step
         print("E-Step, iteration:",I)
         optimal_phi , optimal_gamma = e_step(optimal_beta,initial_alpha, docs,K) 
         print("M-Step, iteration:",I)
         #M-step
         optimal_beta , optimal_alpha = m_step(optimal_phi, optimal_gamma, initial_alpha, V, docs)
         

         if debug:
             print('PHI:',optimal_phi[0])
             print('gamma', optimal_gamma[0,])
             print('alpha:',optimal_alpha)
             print('beta:',optimal_beta[:,0])

         if ( I >= max_iter):
             print('CONVERGED')
             converged = True
         I+=1
    return   optimal_beta , optimal_alpha,  optimal_phi , optimal_gamma


nb_iter=30
optimal_beta , optimal_alpha,  optimal_phi , optimal_gamma =run_em(docs,K, nb_iter,V)



#print top topics

beta_sorted = np.argsort(optimal_beta, axis = 1)[:,V-20:]
for topic in range(K):
    print("\n words for topic number",topic)
    for word in beta_sorted[topic]:
        print(Vocabulary[word],end =' ')
        
        
             
#arange list for presentation
N_top_words =10
words_presentation = ['' for i in range(10)]


for topic in  (0,1,4,6,7):
    for i in range(N_top_words):
        words_presentation[i]  = words_presentation[i] + ' & ' + Vocabulary[beta_sorted[topic][i]]

for i in range(N_top_words):
    words_presentation[i]= words_presentation[i][2:] + '\\\\'
    print(words_presentation[i])