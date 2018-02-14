import numpy as np
import re



class MAPSpamDetector(object):

    def __init__(self):
        self.classes = np.arange(0,2)
        self.word_to_key = {}
        self.key_to_word = [None]*80000
        self.counter = 0
        self.key_counter = 0
        self.X = np.zeros(1000000, dtype=np.int32)
        self.Y = np.zeros(1000000, dtype=np.int32)
        self.log_likelihood = None
        self.log_prior = None

    def _add_word_to_map(self, word, label):
        if not word in self.word_to_key:
            key = self.key_counter
            self.word_to_key[word] = self.key_counter
            self.key_to_word[key] = word
            self.key_counter+=1

        key = self.word_to_key[word]
        self.X[self.counter] = key
        self.Y[self.counter] = label
        self.counter+=1

    def _calc_prior(self):
        self.log_prior = np.zeros(2)
        for k in range(self.counter):
            self.log_prior[self.Y[k]] += 1
        self.log_prior /= self.counter
        self.log_prior = np.log(self.log_prior)
        print("Prior: ", self.log_prior)
    def _calc_likelihood(self):
        self.log_likelihood = np.zeros((self.key_counter, 2))
        for i in range(self.counter):
            self.log_likelihood[self.X[i], self.Y[i]] += 1
        
        # Normalising probabilities
        for i in range(self.log_likelihood.shape[0]):
            self.log_likelihood[i,:] /= self.log_likelihood[i,:].sum()

        for j in range(self.log_likelihood.shape[1]):
            self.log_likelihood[:,j] /= self.log_likelihood[:,j].sum()

        self.log_likelihood = np.log(self.log_likelihood)
        if np.any(self.log_likelihood > 0):
            raise Exception("probabilities not normalised")

        self.log_likelihood[:, 0] -= self.log_prior[0]
        self.log_likelihood[:, 1] -= self.log_prior[1]


    def infer(self, email):
        log_likelihood = np.zeros(2)
        email_words = self._extract_words(email)
        for w in email_words:
            k = self.word_to_key[w]
            #print("Word ", w, " likelihood: ", self.log_likelihood[k, :])
            log_likelihood += self.log_likelihood[k, :] + self.log_prior
            
        return log_likelihood

    def _extract_words(self,email):
        words = []
        email = re.sub(r"[!.\?()\[\]]", " ", email)
        w = email.split(" ")
        words.extend(w)
        return words


    def train(self, emails, labels):

        for email, label in zip(emails, labels):
            words = self._extract_words(email)
            for word in words:
                self._add_word_to_map(word, label)
        self._calc_prior()
        self._calc_likelihood()



