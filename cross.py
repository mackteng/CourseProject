import os
import sys
import numpy as np
import math
import sklearn
import sklearn.model_selection
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.stem import PorterStemmer 
from prettytable import PrettyTable

smallest = 1e-600

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    np.seterr('ignore')
    new_matrix = np.nan_to_num(input_matrix / row_sums[:, np.newaxis], nan=smallest)
    np.seterr('raise')
    return new_matrix
       
class CrossCollectionComparator(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.collections = []
        self.collections_name = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 

        # p ( w | B )
        self.background_word_model = None

        # Common theme models for each theme j
        self.common_theme_model = None

        # Special theme model for each collection per theme
        self.special_per_collection_theme_model = None
        
        # Theme mixing weights for eawch document
        self.document_topic_prob = None
        
        # Hidden variables        
        self.h_topic_prob = None  # P(j | d, c, w)
        self.h_background_prob = None # P(B | d, c, w)
        self.h_common_theme_prob = None # P(C | d, c, w, j)

        # Convenience variables
        self.number_of_collections = 0
        self.total_number_of_documents = 0
        self.vocabulary_size = 0
        self.max_collection_size = 0

    def build_crossCollectionComparator(self):
        """
        Read documents for each collection
        """
        words = set(nltk.corpus.words.words())
        for filename in os.listdir(self.documents_path):
            with open(os.path.join(self.documents_path, filename), 'r') as f:
                lines = f.readlines()
                collection = []
                for line in lines:
                    tokens = word_tokenize(line)
                    tokens = [w.lower() for w in tokens if w in words and len(w) > 4]
                    collection.append(tokens)

                self.collections.append(collection)
                self.collections_name.append(filename.split(".txt")[0])
                # Update count
                self.number_of_collections = self.number_of_collections + 1
                self.total_number_of_documents += len(collection)
                self.max_collection_size = max(self.max_collection_size, len(collection))

    def build_vocabulary(self):
        baseSet = set([])
        for collection in self.collections:
            for document in collection:
                baseSet = baseSet.union(document)
        self.vocabulary = list(baseSet)
        self.vocabulary_size = len(self.vocabulary)

    def build_term_doc_matrix(self):
        def generateRow(line):
            mapping = dict.fromkeys(self.vocabulary, 0)
            for word in line:
                mapping[word] += 1
            return np.array(list(mapping.values()))

        self.term_doc_matrix = np.zeros((self.number_of_collections, self.max_collection_size, self.vocabulary_size))
        for i in range(self.number_of_collections):
            for d in range(len(self.collections[i])):
                self.term_doc_matrix[i][d] = generateRow(self.collections[i][d])

    def initialize_randomly(self, number_of_topics):
        # Special theme model for each collection per theme
        self.special_per_collection_theme_model = np.random.random_sample((self.number_of_collections, number_of_topics, self.vocabulary_size))
        for i in range(self.number_of_collections):
            self.special_per_collection_theme_model[i] = normalize(self.special_per_collection_theme_model[i])

        # Theme mixing weights for each document
        self.document_topic_prob = np.random.random_sample((self.number_of_collections, self.max_collection_size, number_of_topics))
        for i in range(self.number_of_collections):
            self.document_topic_prob[i] = normalize(self.document_topic_prob[i])
        
        # Common theme models for each theme j
        self.common_theme_model = np.random.random_sample((number_of_topics, self.vocabulary_size))
        self.common_theme_model = normalize(self.common_theme_model)

    def expectation_step(self, number_of_topics, param_b, param_c):
        #print("E step:")
        #h_topic_prob
        for i in range(len(self.collections)):
            for d in range(len(self.collections[i])):
                for w in np.nonzero(self.term_doc_matrix[i][d])[0]:
                    row = []
                    for k in range(number_of_topics):
                        try:
                            val = self.document_topic_prob[i][d][k] * ((param_c * self.common_theme_model[k][w])+((1 - param_c) * self.special_per_collection_theme_model[i][k][w]))
                        except:
                            val = smallest
                        row.append(val)
                    np.seterr('ignore')
                    self.h_topic_prob[i][d][w] = np.nan_to_num(np.divide(row, np.sum(row)), nan=smallest)
                    np.seterr('raise')

        #h_background_prob
        for i in range(len(self.collections)):
            for d in range(len(self.collections[i])):
                for w in np.nonzero(self.term_doc_matrix[i][d])[0]:
                    bg = param_b * self.background_word_model[w]
                    nonbg = 0.0
                    for k in range(number_of_topics):
                        try:
                            nonbg += self.document_topic_prob[i][d][k] * (param_c * self.common_theme_model[k][w] + (1 - param_c) * self.special_per_collection_theme_model[i][k][w])
                        except:
                            nonbg += smallest
                    self.h_background_prob[i][d][w] = bg / (bg + (1 - param_b) * nonbg)   
                
        #h_common_theme_prob
        for i in range(len(self.collections)):
            for d in range(len(self.collections[i])):
                for k in range(number_of_topics):
                    for w in np.nonzero(self.term_doc_matrix[i][d])[0]:
                        try:
                            common = param_c * self.common_theme_model[k][w]
                            noncommon = (1 - param_c) * self.special_per_collection_theme_model[i][k][w]
                            self.h_common_theme_prob[i][d][k][w] = common / (common + noncommon)
                        except:
                            if self.common_theme_model[k][w] < self.special_per_collection_theme_model[i][k][w]:
                                self.h_common_theme_prob[i][d][k][w] = smallest
                            else:
                                self.h_common_theme_prob[i][d][k][w] = 1-smallest

    def maximization_step(self, number_of_topics):
        for i in range(self.number_of_collections):
            for d in range(len(self.collections[i])):
                row = []
                for k in range(number_of_topics):
                    sum = 0.0
                    for w in np.nonzero(self.term_doc_matrix[i][d])[0]:
                        sum += self.term_doc_matrix[i][d][w] * self.h_topic_prob[i][d][w][k]
                    row.append(sum)
                self.document_topic_prob[i][d] = row
            self.document_topic_prob[i] = normalize(self.document_topic_prob[i])

        # update common theme model
        for k in range(number_of_topics):
            self.common_theme_model[k] = np.zeros(self.vocabulary_size)
            for i in range(self.number_of_collections):
                for d in range(len(self.collections[i])):
                    for w in np.nonzero(self.term_doc_matrix[i][d])[0]:
                        try:
                            self.common_theme_model[k][w] += self.term_doc_matrix[i][d][w] * (1 - self.h_background_prob[i][d][w]) * self.h_topic_prob[i][d][w][k] * self.h_common_theme_prob[i][d][k][w]
                        except:
                            self.common_theme_model[k][w] += smallest
        self.common_theme_model = normalize(self.common_theme_model)       

        # update special theme model
        for i in range(self.number_of_collections):
            for k in range(number_of_topics):
                self.special_per_collection_theme_model[i][k] = np.zeros(self.vocabulary_size)
                for d in range(len(self.collections[i])):
                    for w in np.nonzero(self.term_doc_matrix[i][d])[0]:
                        try:
                            self.special_per_collection_theme_model[i][k][w] += self.term_doc_matrix[i][d][w] * (1 - self.h_background_prob[i][d][w]) * self.h_topic_prob[i][d][w][k] * (1 - self.h_common_theme_prob[i][d][k][w])
                        except:
                            self.special_per_collection_theme_model[i][k][w] += smallest
            self.special_per_collection_theme_model[i] = normalize(self.special_per_collection_theme_model[i])

    def calculate_likelihood(self, number_of_topics, param_b, param_c):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        likelihood = 0.0
        for i in range(len(self.collections)):
            for d in range(len(self.collections[i])):
                for w in np.nonzero(self.term_doc_matrix[i][d])[0]:
                    nonbg = 0.0
                    for k in range(number_of_topics):
                        try:
                            nonbg += self.document_topic_prob[i][d][k] * ((param_c * self.common_theme_model[k][w]) + ((1 - param_c) * self.special_per_collection_theme_model[i][k][w]))                       
                        except:
                            nonbg += smallest
                    try:
                        likelihood += self.term_doc_matrix[i][d][w] * (np.log((param_b * self.background_word_model[w]) + ((1 - param_b) * nonbg)))
                    except:
                        likelihood += smallest
        self.likelihoods.append(likelihood)
        return likelihood

    def print_results(self, number_of_topics, common_count, special_count):
        x = PrettyTable()
        common_column = ["\n".join(["" for i in range(0, math.ceil((common_count - 1) / 2))] + ["Common"] + ["" for i in range(0, common_count - 1 - math.ceil((common_count - 1) / 2))]), "--"]
        special_column = lambda collectionCount: "\n".join(["" for i in range(0, math.ceil((special_count - 1) / 2))] + [self.collections_name[collectionCount]] + ["" for i in range(0, special_count - 1 - math.ceil((special_count - 1) / 2))])
        final_column = common_column
        for i in range(self.number_of_collections):
            final_column.append(special_column(i))
            final_column.append("--")
        x.add_column("", final_column)
        for k in range(number_of_topics):
            common = ["\n".join(list(map(lambda x : self.vocabulary[x], self.common_theme_model[k].argsort()[-common_count:][::-1]))), "--"]
            special = []
            for c in range(self.number_of_collections):
                s = "\n".join(list(map(lambda x: self.vocabulary[x], self.special_per_collection_theme_model[c][k].argsort()[-special_count:][::-1])))
                special.append(s)
                special.append("---")
            x.add_column("Cluster " + str(k + 1), common + special)
        print(x)

    def cross_collection(self, number_of_topics, max_iter, epsilon, param_b, param_c):
        # build term-doc matrix
        self.build_term_doc_matrix()

        #init background model
        self.background_word_model = np.zeros([self.vocabulary_size], dtype = np.float64)
        for w in range(self.vocabulary_size):
            for i in range(self.number_of_collections):
                for d in range(len(self.collections[i])):
                    self.background_word_model[w] += self.term_doc_matrix[i][d][w]
        self.background_word_model = np.divide(self.background_word_model, np.sum(self.background_word_model))

        # init hidden variables
        self.h_topic_prob = np.zeros([self.number_of_collections, self.max_collection_size, self.vocabulary_size, number_of_topics], dtype=np.float64)
        self.h_background_prob = np.zeros([self.number_of_collections, self.max_collection_size, self.vocabulary_size], dtype = np.float64)
        self.h_common_theme_prob = np.zeros([self.number_of_collections, self.max_collection_size, number_of_topics, self.vocabulary_size], dtype = np.float64)
        
        # Initialize model
        self.initialize_randomly(number_of_topics)

        # Run the EM algorithm
        current_likelihood = 0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step(number_of_topics, param_b, param_c)
            self.maximization_step(number_of_topics)
            newLikelihood = self.calculate_likelihood(number_of_topics, param_b, param_c)
            if(abs(current_likelihood - newLikelihood) < epsilon):
                break
            current_likelihood = newLikelihood

        self.print_results(number_of_topics, 8, 5)

def main():
    np.seterr('raise')
    documents_path = "data/collections/%s" % sys.argv[1]
    crossCollectionComparator = CrossCollectionComparator(documents_path)  # instantiate crossCollectionComparator
    crossCollectionComparator.build_crossCollectionComparator()
    crossCollectionComparator.build_vocabulary()
    print("Vocabulary size:" + str(len(crossCollectionComparator.vocabulary)))
    print("Collection size:" + str(crossCollectionComparator.number_of_collections))
    print("Number of documents:" + str(crossCollectionComparator.total_number_of_documents))
    print("Max collectionsize:" + str(crossCollectionComparator.max_collection_size))
    number_of_topics = 4
    max_iterations = 1000
    epsilon = 0.001
    param_b = 0.95
    param_c = 0.30
    crossCollectionComparator.cross_collection(number_of_topics, max_iterations, epsilon, param_b, param_c)

if __name__ == '__main__':
    main()
