import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import special
import matplotlib.pyplot as plt
import re
import pandas as pd

#TODO
#REFACTOR!

#set path to the input file folder
n_features = 3000  # build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
K = 100 #num topics
n_top_words = 20  # for prints
max_treshold = 0.95  # high frequency words
min_treshold = 2  # low frequency words
convergence = 0.25 #convergence stopping value for EM
nb_iter = 30  #maximum number of itteration for EM
train_size = 2000 # numer of document in the traininh algorithm

# For debugging
debug = False

# Read data
documents = []

with open('blei_samples.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(r'\n')]
        inner_list = re.sub(r'\d+', '', inner_list[0]) #remove numbers
        documents.append(inner_list)


# Prepare data
cv = CountVectorizer(max_df=max_treshold,
                     min_df=min_treshold,
                     max_features=n_features,
                     stop_words='english')

DOC_train = cv.fit_transform(documents[0:train_size])

DOC_holdout = cv.transform(documents[train_size:])


Vocabulary = [str.strip('_') for str in cv.get_feature_names()]
# get a matrix of counts for each element of vocabulary
Vocab_per_Doc_count_train = DOC_train.toarray()

Vocab_per_Doc_count_holdout = DOC_holdout.toarray()

# Number of Documents M/ Size of Vocabulary V
M, V = DOC_train.shape


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


docs_train = list(map(DataTrans, Vocab_per_Doc_count_train))
docs_holdout = list(map(DataTrans, Vocab_per_Doc_count_holdout))


# ===========================================================================================================
# ===========================================================================================================

def log_gamma(x):
    return special.loggamma(x)


def psi(x):
    '''
    Called di_gamma in the Blei-C  implementation

    '''
    return special.polygamma(0, x)


def psi_2(x):
    '''
    Called tri_gamma in the Blei-C  implementation

    '''
    return special.polygamma(1, x)


def gradient(alpha, gamma):
    '''
    Called d_alhood in the Blei-C  implementation
    see annexe A.4.2 for derivation
    '''
    M = gamma.shape[0]

    ss = sufficient_statistic(gamma)

    D_alpha = M * (psi(alpha.sum()) - psi(alpha)) + ss
    return D_alpha


def sufficient_statistic(x):
    '''
    COmpute the sufficient statistic from the gamma matrix
    '''
    ss = (psi(x) - psi(x.sum(axis=1, keepdims=True))).sum(axis=0)
    return ss


def update_alpha_hessian_trick(alpha, gamma):
    '''
    newton update
    see annexe A.2 for derivation
    '''
    N_doc = gamma.shape[0]

    D1_alpha = gradient(alpha, gamma)

    D2_alpha_diagonal = - psi_2(alpha) * N_doc

    z = psi_2(alpha.sum()) * N_doc

    c = (D1_alpha / D2_alpha_diagonal).sum() / (1 / z + (1 / D2_alpha_diagonal).sum())

    update = (D1_alpha - c) / D2_alpha_diagonal
    return alpha - update


def newton_alpha(alpha, gamma):
    '''
    run newton update until convergence of parameter
    input:
    gamma matrix from the m step

    '''
    I = 0
    converged = False
    optimal_alpha = alpha
    while not converged:

        optimal_alpha_old = optimal_alpha
        optimal_alpha = update_alpha_hessian_trick(optimal_alpha, gamma)
        delta = np.linalg.norm(optimal_alpha - optimal_alpha_old)
        I += 1
        if (delta < 10e-5 or I > 100):
            converged = True
            print('alpha update stoped after:', I, 'iterations, delta:',delta)


    return optimal_alpha


# ==========================================================================================================
# ==========================================================================================================

def e_step(beta_t, alpha_t, docs, K):
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
    M = len(docs)
    N = [doc.shape[0] for doc in docs]
    # Initialization
    optimal_phi = []
    optimal_gamma = np.zeros((M, K))

    # iterate for each document
    for d in range(M):
        doc = docs[d]
        # initialization for  each document
        optimal_phi_doc =  np.ones((N[d], K))/K  # (1)
        optimal_gamma_doc = alpha_t + (N[d]/K)  # (2) 
        converged = False
        I = 0
        while not converged:
            I +=1
            old_optimal_gamma_doc = optimal_gamma_doc
            old_optimal_phi_doc = optimal_phi_doc

            # update phi
            optimal_phi_doc = (doc @ beta_t.T) * np.exp(psi(optimal_gamma_doc) - psi(optimal_gamma_doc.sum()))  # (6)
            optimal_phi_doc = optimal_phi_doc / (np.sum(optimal_phi_doc, axis=1)[:, None])  # (7)
            # update gamma
            optimal_gamma_doc = alpha_t + np.sum(optimal_phi_doc, axis=0)  # (8)
            # check convergence
            delta_gamma_doc = np.linalg.norm(optimal_gamma_doc- old_optimal_gamma_doc)
            delta_phi_doc = np.linalg.norm(optimal_phi_doc - old_optimal_phi_doc)

            if (delta_gamma_doc< 10e-3 and  delta_phi_doc< 10e-3):
                converged = True
                if (d % 100)== 0:
                    print('document:', d, 'coverged after', I,'iterations  gamma:', delta_gamma_doc,'phi:',  delta_phi_doc)

        optimal_phi.append(optimal_phi_doc)
        optimal_gamma[d] = optimal_gamma_doc
    return optimal_phi, optimal_gamma


def m_step(phi_t, gamma_t, initial_alpha, V, docs):
    '''

    inputs:
       phi_t : phi paramters from the E-step, shape = [M x N[d] x K] (vary per document)
       gamma_t: matrix of gamma parameter from the E-step,  shape  = M x K
       initial_alpha: matrix of gamma parameter from the E-step,  shape  =N document X N topics
       V: n_features
       words: list of  list of word index present in each of the document ,shape = M document X N[d] words  (vary per document)

    an iteration computation of optimal beta and alpha
       Output:

    beta   : [K x V]
    alpha  : dirichlet parameter [K]

    '''
    # initialization
    M, K = gamma_t.shape
    optimal_beta = np.zeros((K, V))
    optimal_alpha = np.zeros((1, K))

    # update beta
    beta_per_doc = np.zeros((M, K, V))
    for d in range(M):
        beta_per_doc[d] = phi_t[d].T @ docs[d]

    optimal_beta = np.sum(beta_per_doc, axis=0)
    # Normalization of beta
    optimal_beta = optimal_beta / (np.sum(optimal_beta, axis=1))[:, np.newaxis]

    if debug:
        print("The next array should contain only ones")
        print(np.sum(optimal_beta, axis=1))
        print(optimal_alpha.shape)

    # update alpha : we use Newton Raphson method to update alpha
    optimal_alpha = newton_alpha(initial_alpha, gamma_t)

    return optimal_beta, optimal_alpha


# =============================================================================================================
# =============================================================================================================


def run_em(docs, K, max_iter,stoping_metric, V ):
    '''
    run the E-step and M-Step iteratively until the log likelihood converges
    returns the final parameters

    inputs:
        N: number of words in each document
        Doc_words: list of  list of word index present in each of the document ,shape = M document X N words
        Initial_alpha: initial alpha parameters used
        V: n_features
    '''

    # initialisation
    optimal_beta = np.random.dirichlet(np.ones(V), K)  
    initial_alpha = np.random.gamma(shape=np.ones((K)), scale=1/K)
    
    gamma_old = np.zeros((M, K))
    converged = False
    I = 0
    delta_list= []

    # Run EM Algorithm
    while not converged:
        I += 1
        # E-step
        print("E-Step, iteration:", I)
        optimal_phi, optimal_gamma = e_step(optimal_beta, initial_alpha, docs, K)
        print("M-Step, iteration:", I)
        # M-step
        optimal_beta, optimal_alpha = m_step(optimal_phi, optimal_gamma, initial_alpha, V, docs)

        if debug:
            print('PHI:', optimal_phi[0])
            print('gamma', optimal_gamma[0,])
            print('alpha:', optimal_alpha)
            print('beta:', optimal_beta[:, 0])

        delta = np.sqrt(np.mean(np.square(optimal_gamma-gamma_old)))
        delta_list.append(delta)
        
        print("delta_gamma:", delta, 'iteration:', I) 
        gamma_old=optimal_gamma
        if (I >= max_iter or delta <= convergence):
            print('CONVERGED')
            converged = True
        
        

    return optimal_beta, optimal_alpha, optimal_phi, optimal_gamma,delta_list

#run em 
optimal_beta, optimal_alpha, optimal_phi, optimal_gamma, delta_list= run_em(docs_train, K, nb_iter,convergence, V)

# =============================================================================================================
#extract results for report an presentation




# extract  top words from each  topics
beta_sorted = np.argsort(optimal_beta, axis=1)[:, V - 10:] #sort beta
top_words_per_topics = []
for topic in range(K):
    top_words_per_topics.append([Vocabulary[word] for word in beta_sorted[topic]])
        
#save
top_words_per_topics_table =pd.DataFrame(top_words_per_topics)
top_words_per_topics_table.to_csv("C:/Users/rober/OneDrive/Bureau/etude/graph models udem/Projet/ift6269-topic-modeling/rapport/implementation/TopicsList.csv")
  
#predict on new document
phi_holdout, gamma_holdout = e_step(optimal_beta, optimal_alpha, docs_holdout, K)

#predictions_holdout = (gamma_holdout/gamma_holdout.sum(axis=1,keepdims =True)).max(axis=1)
topic_holdout =gamma_holdout.argmax(axis=1)



output_holdout = pd.DataFrame({
                                "text": documents[train_size:],
                                "topic_id" : topic_holdout , 
                                "prediction": predictions_holdout
    
                                })


output_holdout = pd.merge(output_holdout,  pd.DataFrame(top_words_per_topics_table),left_on = 'topic_id', right_index=True)

#save
output_holdout.to_csv("C:/Users/rober/OneDrive/Bureau/etude/graph models udem/Projet/ift6269-topic-modeling/rapport/implementation/classification_holdout.csv")

#create  chart to illustrate  convegence
%matplotlib inline
plt.style.use('ggplot')




y = delta_list

x_pos = [i for i, _ in enumerate(range(len(delta_list)))]
f = plt.figure()
plt.bar(x_pos, y, color='green')
plt.xlabel("Iterations")
plt.ylabel("L2 Delta")

plt.xticks(x_pos)
plt.plot([0 ,len(delta_list)],[convergence, convergence] )


plt.show()

#arange list for latex presentation
N_top_words =10
words_presentation = ['' for i in range(10)]


for topic in (1,3,27,87,67):
    for i in range(N_top_words):
        words_presentation[i]  = words_presentation[i] + ' & ' + Vocabulary[beta_sorted[topic][i]]

for i in range(N_top_words):
    words_presentation[i]= words_presentation[i][2:] + '\\\\'
    print(words_presentation[i])
