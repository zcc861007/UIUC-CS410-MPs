import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 

        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        f = open(self.documents_path)

        for line in f:
          line = line.strip()
          line = line.split()
          self.documents.append(line)
        
        self.number_of_documents = len(self.documents)
        # #############################
        
        # pass    # REMOVE THIS

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        self.vocabulary = list(set([item for sublist in self.documents for item in sublist]))
        self.vocabulary_size = len(self.vocabulary)
        # #############################
        
        # pass    # REMOVE THIS

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here

        # c_il, i: doc, lw: word
        self.term_doc_matrix = np.zeros(shape=(self.number_of_documents,
                                               self.vocabulary_size))
        for (i, doc) in enumerate(self.documents):
          for (lw, word) in enumerate(self.vocabulary):
            self.term_doc_matrix[i,lw] = doc.count(word)
        # ############################
        
        # pass    # REMOVE THIS


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here

        # pi_i,j, i: doc, j: topic
        self.document_topic_prob = np.random.random(size=[self.number_of_documents,
                                                          number_of_topics])
        self.document_topic_prob = normalize(self.document_topic_prob)

        # theta_j,lw, j: topic, lw: word
        self.topic_word_prob = np.random.random(size=[number_of_topics,
                                                      self.vocabulary_size])
        self.topic_word_prob = normalize(self.topic_word_prob)
        # ############################

        # pass    # REMOVE THIS
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        for i in range(self.number_of_documents):
          p_z = self.document_topic_prob[i,:][..., None] * self.topic_word_prob  # (j, 1) * (j, lw)
          p_z = normalize(p_z.T).T
          self.topic_prob[i,:,:] = p_z
        
        # topic_prob = []
        # for i in range(self.number_of_documents):
        #   p_z = self.document_topic_prob[i,:] * self.topic_word_prob.T  # (j,) * (j, lw).T
        #   p_z = normalize(p_z)
        #   topic_prob.append(p_z)
        # self.topic_prob = np.array(topic_prob)  # (i, lw, j)
        # ############################

        # pass    # REMOVE THIS
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        for j in range(number_of_topics):
          # theta = np.sum(self.term_doc_matrix * self.topic_prob[:,:,j], axis=0)   # (i, lw) * (i, lw, j=c)
          theta = np.sum(self.term_doc_matrix * self.topic_prob[:,j,:], axis=0)   # (i, lw) * (i, j=c, lw)
          self.topic_word_prob[j,:] = theta
        self.topic_word_prob = normalize(self.topic_word_prob)
        # ############################
        
        # update P(z | d)

        # ############################
        # your code here
        for i in range(self.number_of_documents):
          # pi = self.term_doc_matrix[i,:] @ self.topic_prob[i,:,:]   # (i=c, lw) @ (i=c, lw, j)
          # pi = self.topic_prob[i,:,:] @ self.term_doc_matrix[i,:]  # (i=c, j, lw) @ (i=c, lw)
          pi = np.matmul(self.topic_prob[i,:,:], self.term_doc_matrix[i,:])  # (i=c, j, lw) @ (i=c, lw)
          self.document_topic_prob[i,:] = pi
        self.document_topic_prob = normalize(self.document_topic_prob)
        # ############################
        
        # pass    # REMOVE THIS


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # logL = np.sum(self.term_doc_matrix * np.log(self.document_topic_prob @ self.topic_word_prob))  # (i, lw) * ((i, j) @ (j, lw))
        logL = np.sum( self.term_doc_matrix * np.log(np.matmul(self.document_topic_prob, self.topic_word_prob)) )  # (i, lw) * ((i, j) @ (j, lw))
        self.likelihoods.append(logL)
        # ############################
        
        return logL

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        # self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size])

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            self.expectation_step()
            self.maximization_step(number_of_topics)
            current_likelihood = self.calculate_likelihood(number_of_topics)
            print('current_likelihood: {:.3f}'.format(current_likelihood))

            if iteration == max_iter-1:
              print('==============================')
              print('EM iterations are done!')
              
              # print('vocabulary =', self.vocabulary)
              print('theta_j,lw =')
              # print(self.topic_word_prob.round(3))
              print(np.concatenate([np.array(self.vocabulary)[None, ...], self.topic_word_prob.round(3)], axis=0))
              
              print('pi_i,j =')
              print(self.document_topic_prob[:10].round(3))
            # ############################

            # pass    # REMOVE THIS



def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
