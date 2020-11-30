import numpy as np
import Cupy as cp
import random

from Utils import Utils
import settings.py

"""
follow the  blei implementation in C: 
https://github.com/blei-lab/lda-c    
with the structure of scikit learn library in python
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/_lda.py#L31
"""

class LatentDirichletAllocation():
    """
    
    Latent Dirichlet Allocation 
     
    Parameters
    ----------
    num_topics : int,  Number of topics.
    
    gamma : float,  Prior of document topic distribution  (theta in the paper) 
    
    phi : float, Prior of topic word distribution (beta in the paper)
    
    
    max_iter 
    
    Attributes
    ----------
    """


    def __init__(self, num_topics=10,
                 gamma=None,
                 max_iter=10,
                 evaluate_every=-1,
                 mean_change_tol=1e-3, max_doc_update_iter=100,
                 n_jobs=None, verbose=0, random_state=None):
        
    self.num_topics = num_topics
    self.gamma = gamma
    self.phi = phi
    self.max_iter = max_iter
    self.NEWTON_THRESH = 1e-5
    self.MAX_ALPHA_ITER = 1000 
    self.
    
    ###section 1: LdaAlpha ###
    
    # Objective function L: double a, double ss, int D, int K
    def alhood(self, a, ss, D, K):
        factor = Utils.log_gamma(K * a) - Utils.log_gamma(a)
        return D * factor + (a - 1) * ss

    # First derivative of L: double a, double ss, int D, int K
    def d_alhood(self, a, ss, D, K):
        factor = (K * Utils.di_gamma(K * a) - K * Utils.di_gamma(a))
        return D * factor + ss

    # Second derivative of L: double a, int D, int K
    def d2_alhood(self, a, D, K):
        factor = (K * K * Utils.tri_gamma(K * a) - K * Utils.tri_gamma(a))
        return D * factor

    # Implement Newton's method
    def opt_alpha(self, ss, D, K):
        a, log_a, init_a = 100
        iter = 0
        while True:
            iter += 1
            a = np.exp(log_a)
            if np.isnan(a):
                init_a = init_a * 10
                print("WARNING: alpha is NaN. New init = ", init_a)
                a = init_a
                log_a = np.log(a)
            f = self.alhood(a, ss, D, K)
            df = self.d_alhood(a, ss, D, K)
            d2f = self.d2_alhood(a, D, K)
            log_a = log_a - (df / (d2f * a + df))
            print("Maximisation of alpha: %s  %s" % (f, df))

            if np.abs(df) > self.NEWTON_THRESH and iter < self.MAX_ALPHA_ITER:
                break

        return np.exp(log_a)
    
    ###section 2: utlity functions###

    def log_gamma(self, x):
        return math.lgamma(x)
    
    def tri_gamma(self, x):
        return special.polygamma(2, x)
    
    def di_gamma(self, x):
        return special.polygamma(1, x)

    ###section 3: innitialisation and mle###

    # compute MLE lda model from sufficient statistics
    def lda_mle(model, ss, estimate_alpha):       # lda_model model, lda_suffstats ss, int estimate_alpha
      for k in range(model.num_topics):
        for w in range(model.num_terms):
          if ss.class_word[k][w] > 0:
            model.log_prob_w[k][w] = cp.log(ss.class_word[k][w]) - cp.log(ss.class_total[k])
          else:
            model.log_prob_w[k][w]= -100
      if estimate_alpha == 1:
        model.alpha = opt_alpha(ss.alpha_suffstats,ss.num_docs,model.num_topics)
        print("new alpha = %.5f" %model.alpha)
    
    # allocate sufficient statistics
    def new_lda_suffstats(model):                 # lda_model model
      num_topics = model.num_topics
      num_terms = model.num_terms
      ss = lda_suffstats()
      ss.class_total = cp.zeros(num_topics)
      ss.class_word = cp.zeros((num_topics,num_terms))
      return ss
    
    # various initialisations for the sufficient statistics
    def zero_initialize_ss(ss, model):
      ss.class_total = cp.zeros(model.num_topics)
      ss.class_word = cp.zeros((model.num_topics,model.num_terms))
      ss.num_docs = 0
      ss.alpha_suffstats = 0
    
    def random_initialize_ss(ss, model): 
      num_topics = model.num_topics
      num_terms = model.num_terms
      class_word = cp.random.rand(num_topics,num_terms)
      class_total = cp.sum(class_word, axis = 1)
      ss.class_word = class_word
      ss.class_total = class_total
    
    def corpus_initialize_ss(ss, model, c):
      num_topics = model.num_topics
      doc = document()
      seen = []
      already_selected = True
      for k in range(num_topics):
        while already_selected:
          d = random.randint(0, c.num_docs-1)
          already_selected = False
          for j in range(k):
            if seen[j] == d:
              already_selected = True
              print("skipping duplicate seed document" + d)
        seen[k] = d 
        print("initialized with document" +d)
        doc = c.docs[d]
        for n in range(doc.length):
          ss.class_word[k][doc.word[n]] += doc.counts[n]
        
        for n in range(model.num_terms):
          ss.class_word[k,n] += 1
          ss.class_total[k] = ss.class_total[k] + ss.class_word[k,n]
    
    
    def manual_initialize_ss(seedfile, ss, model, c):
    
    # allocate new lda model
    def new_lda_model(num_terms, num_topics):
      model = lda_model()
      model.num_topics = num_topics
      model.num_terms = num_terms
      model.alpha = 1
      model.log_prob_w = cp.zeros((num_topics,num_terms))
      return model
    
    # save an lda model
    def save_lda_model(model, model_root):
    
    def load_lda_model(model_root):
    
        
     ###section 4: parameter estimation and inference###
      
    def doc_e_step(doc,gamma, phi, model,ss):
        """
        compute the E-step of the algorithm.
        perform inference on a document and update sufficient statistics
        
        Params:
        doc: a document from the corpus
        Gamma: The variational Dirichlet parameters for each document
        phi: the word distribution for topic k
        model: lda_model class
        ss: lda_suffstats class
          
        Return:  likelihood    
        """
        likelihood  =model.lda_inference(doc)
    
        #update sufficient statistics
        gamma_sum = 0
        for k in range(model.num_topics):#loop on every topic
           gamma_sum += gamma[k]
           ss.alpha_suffstats += digamma(gamma[k])
           
        ss.alpha_suffstats -=  model.num_topics *digamma(gamma_sum)
        
        for n in  doc:  #loop on every word
            for k in range(model.num_topics): #loop on every topic
                ss.class_word[k][n] +=  counts[n]*phi[n][k] 
                ss.class_total=  counts[n]*phi[n][k]
        ss.num_docs +=1 #RC: not sure this is usefull
        for doc in document:
        
        
        return(likelihood)

     #### run EM####
     
      def run_em( start, directory,  corpus, INITIAL_ALPHA):
          """
       #  run EM
       Params:
        -start: seeded or random
        -directory: where to save the outut (  i think we can change this part)
        -INITIAL_ALPHA:  from the setting file 
        -corpus: corpus object
        Return:  lda model???  
        """
     
          #initialize model
          
       
        if start !="seeded":
            model = lda_model.new_lda_model(corpus.num_terms, NTOPICS)
            ss = lda_model.new_lda_suffstats(model)
            lda_model.corpus_initialize_ss(ss, model, corpus)
            lda_model.lda_mle(model, ss, 0)
            model.alpha = INITIAL_ALPHA
            
        elif  start != "random":# here dont give the corpus as an argument
            model = lda_model.new_lda_model(corpus.num_terms, NTOPICS)
            ss = lda_model.new_lda_suffstats(model)
            lda_model.corpus_initialize_ss(ss, model)
            lda_model.lda_mle(model, ss, 0)
            model.alpha = INITIAL_ALPHA
        else: #use an already existing model
            model = lda_model.new_lda_model(start)
            
    
    
        # run expectation maximization
        i=0
        likelihood_old = 0
        converged = 1
        
        while (converged < 0) | (converged > EM_CONVERGED) | (i <= 2)) & (i <= EM_MAX_ITER)):
            i+=1
            print("em iteration:",i)
            likelihood = 0
            zero_initialize_ss(ss, model)
            #E-step
            for d, doc in enumerate(corpus):
                print("document:" ,d)
                
                likelihood += doc_e_step(doc),
                                         var_gamma[d],
                                         phi,
                                         model,
                                         ss)
                
            #M-Step
            lda_mle(model, ss, ESTIMATE_ALPHA)
            
            # check for convergence
            converged = (likelihood_old - likelihood) / (likelihood_old)
            if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2
            likelihood_old = likelihood
            
        #output model and likelihood
        #dont this this is necessairy, we can  return these value instead
        
    
      def infer(model, corpus):
               """
       #  do prediction on unseen  data
       Params:
        -model: fitted model used to predict on new corpus
        -corpus: the corpus to be scored
        Return: likelihood 
        """
        likelihood =np.zeros( len(corpus))
        for d, doc in  enumerate(corpus):
            	likelihood[d] = lda_inference(doc, model, var_gamma[d], phi)
                print("likelihood document:",d ,":" ,likelihood[d]  )
            
        return likelihood
