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

os.chdir('C:/Users/rober/OneDrive/Bureau/etude/graph models udem/Projet/ift6269-topic-modeling/latent-dirichlet-allocation/data')
#inputs; 
n_features = 1000 #build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
n_components = 10 #Number of topics.
n_top_words = 20 #for prints
max_treshold = 0.95 # high frequency words
min_treshold = 2 #low frequency words

# For debugging
debug = False

#Read data
documents = []

with open('samples.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split( r'\n')]
        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        documents.append(inner_list[0])
        



#Prepare data
cv = CountVectorizer(max_df=max_treshold, min_df=min_treshold,
                       max_features=n_features,
                       stop_words='english')


DOC = cv.fit_transform(documents)

Vocabulary = [str.strip('_') for str in cv.get_feature_names()]
#get a matrix of counts for each element of vocabulary
Vocab_per_Doc_count = DOC.toarray()

# Number of Documents M/ Size of Vocabulary V 
M , V = DOC.shape
K = 100


if debug:
  print("shape of matrix counts for (doc,word)= "+ str(Vocab_per_Doc_count.shape))
  print("size of vocabulary set = "+ str(len(Vocabulary)))

#lemmatizationnot ready yet
# =============================================================================
# documents = clean_data(documents)
# documents = remove_stop_words(documents)
# documents = remove_empty(documents)
# 
# 
# # Lemmatization
# from nltk import word_tokenize          
# from nltk.stem import WordNetLemmatizer 
# class LemmaTokenizer:
#   def __init__(self):
#     self.wnl = WordNetLemmatizer()
#   def __call__(self, doc):
#     return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
# 
#         
# =============================================================================
  
  
  
  
#to do: check if needed
  ''' 
Doc_words :list of words for each document d (words are represented by their index in Vocabulary) [M x N[d]]
N :list of size of each document (number of words)
'''
Doc_words = []
N = []
for d in range(M):
  words_list = []
  for vocab, count in enumerate(Vocab_per_Doc_count[d]):
    if count!=0:
      words_list.extend([vocab for i in range(count)]) 

  # add the list of words for document d
  Doc_words.append(words_list)
  # add the number of words in document d
  N.append(len(words_list))
# generators
#print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) 


  
#Parameter estimation
#functions

def log_gamma( x):
        return  special.loggamma(x)
    
def psi( x):
    '''
    Called di_gamma in the Blei-C  implementation
    
    '''
    return special.polygamma(1, x)


def psi_2( x):
    '''
    Called tri_gamma in the Blei-C  implementation
    
    '''
    return special.polygamma(2, x)


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
    
    D2_alpha_diagonal=  psi_2(alpha)*N_doc
    
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
      
      optimal_alpha_old= optimal_alpha
      optimal_alpha =update_alpha_hessian_trick(optimal_alpha ,gamma)
      I +=1
      delta = np.linalg.norm(optimal_alpha- optimal_alpha_old) 


      if (delta < 10e-3 or I >100):
          converged = True   
          print('stoped after:',I,'iterations')
          
    return optimal_alpha
	
# =============================================================================
# def lda_compute_likelihood(self, doc, lda_model, phi, var_gamma):
#   #counts: number of occurences for nth word in counts
#   #words: nth word of vocabulary
#   likelihood = 0
#   digsum = 0
#   var_gamma_sum = 0
#   K = lda_model.num_topics
#   alpha = lda_model.alpha
#   N = doc.length
# 
#   #Initialize
#   dig = digamma(var_gamma)
#   var_gamma_sum = np.sum(var_gamma)
#   digsum = digamma(var_gamma_sum)
# 
#   l1 = lgamma(alpha * K)
#   l2 = (K * lgamma(alpha))
#   l3 = lgamma(var_gamma_sum)
#   likelihood =  l1 - l2 - l3
# 
#   for k in range(K):
#     ll1 = (alpha-1) * (dig[k] - digsum)
#     ll2 = lgamma(var_gamma[k])
#     ll3 = (var_gamma[k] - 1) * (dig[k] - digsum)
#     likelihood += ll1 + ll2 - ll3
#   
#     for n in range(N):
#       if phi[n][k] > 0:
#           lll1 = phi[n][k] * ((dig[k] - digsum) - np.log(phi[n][k]))
#           lll2 = doc.counts[n]* ll1 + lda_model.log_prob_w[k, doc.words[n]]
#   
#   return likelihood
# =============================================================================
def lda_compute_likelihood( doc, phi, var_gamma,alpha,beta,V):
    """
    compute the log likelihood  of 1 document
    
    input:
    doc : list of word index present in the document ,shape = n words 
    phi: matrix of phi parameter in the document, shape  =n words X n topics
    var_gamma: vector of gamma parameter for the document, shape  = n topics
    alpha:  vector of gamma parameter for the document, shape  = n topics 
    beta : matrix of beta   parameter for the document , shape = n topic X n_features !!!double check if this is okk!!!
    V: n_features
    """
    
    #innitialization
    likelihood = 0
    digsum = 0
    var_gamma_sum = 0
    N = len(doc)
    
    #compute each term of log likelihood
    dig = psi(var_gamma)
    var_gamma_sum = np.sum(var_gamma)
    digsum = psi(var_gamma_sum)
    
    l_alpha_1 = log_gamma(alpha.sum())
    l_alpha_2 = log_gamma(alpha).sum()
    
    l_gamma_1 = log_gamma(var_gamma_sum)
    l_gamma_2 = log_gamma(var_gamma).sum()
    l_gamma_3 = ((dig-1) * (dig - digsum)).sum()
    
    l_alpha_gamma = ((alpha-1) * (dig - digsum)).sum()
    
    l_phi = (np.log(phi)*phi).sum()
    
    l_phi_gamma = (phi* (dig-digsum)).sum()
    
    #to to check
    w = np.zeros((N,V))
    w[np.arange(N), doc] = 1
    l_phi_beta = (np.matmul(phi.T,w)*beta).sum()
    
    #sum the loglikelihood
    likelihood = l_alpha_1 - l_alpha_2 + l_alpha_gamma + l_phi_gamma  -l_gamma_1 +l_gamma_2 -l_gamma_3 - l_phi_beta-l_phi
    
    return likelihood

#test =lda_compute_likelihood(Doc_words[0],optimal_phi[0], optimal_gamma[0], optimal_alpha,optimal_beta, V   )




def e_step(beta_t ,alpha_t ,N ,words):
  '''
  Input:
      beta_t: K x V matrix
      alpha_t: M x K matrix
      N: M X 1 list with the length of each document
      
  
  an iteration computation of optimal phi and gamma
  phi   : variational multinomial_parameters [M x N[d] x K]
  gamma : variational dirichlet parameter [M x K]
  N  : list of number of words in each document
  '''
  
  M = len(N)
  K = alpha_t.shape[1]
  
  #Innitialization
  optimal_phi = []
  optimal_gamma = np.zeros((M,K))
  
  #itterate for each document
  for d in range(M):

    #initialization for  each document
    optimal_phi_doc = 1/K * np.ones((N[d],K))#(1)
    optimal_gamma[d] += alpha_t[d]+ np.max((N[d]/K,0.2))  #(2) added a minimum value so that the spi doesnt create overflow
    converged = False

    while not converged:

      old_optimal_gamma = optimal_gamma[d]
      old_optimal_phi_doc = optimal_phi_doc

      # update phi
      for n in range(N[d]): #(4)

        #optimal_phi_doc[n] = np.multiply(beta_t[:,words[d][n]], np.exp( psi(optimal_gamma[d]))) #(6)
        optimal_phi_doc[n] = np.multiply(beta_t[:,words[d][n]], np.exp(psi(optimal_gamma[d])-psi(optimal_gamma[d].sum()))) #(6)
  
        
        optimal_phi_doc[n]= optimal_phi_doc[n] / np.sum(optimal_phi_doc[n]) #(7)
      
      # update gamma
      optimal_gamma[d] = alpha_t[d]  + np.sum(optimal_phi_doc,axis = 0) # (8)
      # check convergence
      if (np.linalg.norm(optimal_gamma[d] - old_optimal_gamma) < 10e-3 and np.linalg.norm(optimal_phi_doc - old_optimal_phi_doc) < 10e-3):
        converged = True
    optimal_phi.append(optimal_phi_doc)
    print('document:', d)
    print('optimal_phi_doc:',optimal_phi_doc)


  return optimal_phi,optimal_gamma  




def m_step(phi_t ,gamma_t ,Innial_alpha, V ,words, N):
  '''
  an iteration computation of optimal beta and alpha
  beta   : [K x V]
  alpha  : dirichlet parameter [M,K]
  words:  list of words in each documents
  
  inputs:
     phi_t : phi paramters from the E-step, shape = [M x N[d] x K] (vary per document)
     gamma_t: matrix of gamma parameter from the E-step,  shape  =N document X N topics
     Innial_alpha: matrix of gamma parameter from the E-step,  shape  =N document X N topics
     V: n_features
     words: list of  list of word index present in each of the document ,shape = N document X N words  (vary per document)
     
  '''
  #innitilization
  M, K = gamma_t.shape
  optimal_beta = np.zeros((K,V))
  optimal_alpha=  np.zeros((Innial_alpha.shape)) 

  # update beta
  beta_per_doc = np.zeros((M,K,V))
  for d in range(M):

    # w = One hot encoding of words 
    w = np.zeros((N[d],V))
    w[np.arange(N[d]),words[d]] = 1

    beta_per_doc= np.matmul(phi_t[d].T,w)

  optimal_beta = np.sum(beta_per_doc, axis=0)
  #Normalization of beta  
  optimal_beta = optimal_beta / np.expand_dims(np.sum(optimal_beta, axis= 1), axis=1)

  if debug:
    print(optimal_beta.shape)
    print("The next array should contain only ones")
    print(np.sum(optimal_beta,axis = 1))
    
  # update alpha : we use Newton Raphson method one document at the time
  for d in range(M):
      optimal_alpha[d]= newton_alpha(Innial_alpha[d],gamma_t)
      
  return optimal_beta,optimal_alpha
    

def run_em(N,Doc_words,Innial_alpha,V):
    '''
    run the E-step and M-Step itteratively until the log likelihood converges
    returns the final parameters
    
    inputs:
        N: number of words in each document 
        Doc_words: list of  list of word index present in each of the document ,shape = N document X N words  
        Innial_alpha: innital alpha parameters used
        V: n_features
    '''

    #innitialisation
    optimal_beta = 1/V * np.ones((K,V))
    optimal_alpha = Innial_alpha
    likelihood=-10000000000
    converged = False
    I = 0
    
    #Run EM Algorithm
    while not converged:
         likelihood_old =likelihood

         #E-step
         print("E-Step, iteration:",I)
         optimal_phi , optimal_gamma = e_step(optimal_beta,optimal_alpha, N,Doc_words) 
         print("M-Step, iteration:",I)
         #M-step
         optimal_beta , optimal_alpha = m_step(optimal_phi, optimal_gamma, Innial_alpha, V, Doc_words, N)
         
         #compute likelihood
         likelihood = 0
         for J in range(len(Doc_words)):
             likelihood += lda_compute_likelihood(Doc_words[J],optimal_phi[J], optimal_gamma[J], optimal_alpha,optimal_beta, V   )
         
         #compare previous iteration likelihood with current itteration, stop if converged   
         ll_delta = (likelihood_old - likelihood) / (likelihood_old)
         print('log likelihood before:',likelihood_old)
         print('log likelihood after:',likelihood)
         print('log likelihood change:',ll_delta)
         if (ll_delta < 10e-3  or I >100):
             print('CONVERGED')
             converged = True
         I+=1
    return   optimal_beta , optimal_alpha,  optimal_phi , optimal_gamma



Innit_alpha = np.ones((M,K))*1/K
optimal_beta , optimal_alpha,  optimal_phi , optimal_gamma =run_em(N,Doc_words,Innit_alpha,V)

#optimal_phi , optimal_gamma = e_step(beta_t,alpha_t,N,Doc_words) 

#optimal_beta , optimal_alpha = m_step(optimal_phi, optimal_gamma, V, Doc_words, N)
