""" Contains the part of speech tagger class. """
import math
import collections
import nltk
from statistics import stdev
import csv

//
def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    sentence_fileObj = open(sentence_file, "r")
    words = sentence_fileObj.read().splitlines()
    words_data = []
    for word in words:
        word = word.split(",",1)
        if word[0] == "id":
            continue
        word[1] = word[1].strip('\"').lower()
        words_data.append(word[1])
    if tag_file is None:
        return words_data
    tags_fileObj = open(tag_file, "r")
    tags = tags_fileObj.read().splitlines()
    data = []
    for i in range(len(tags)):
        tag = tags[i].split(",",1)
        if tag[0] == "id":
            continue
        pair = []
        pair.append(words_data[i-1])
        tag[1] = tag[1].strip('\"')
        pair.append(tag[1])
        data.append(pair)
    return data


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy

    You might want to refactor this into several different evaluation functions.

    """
    pass


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        self.lambda1 = 0.0
        self.lambda2 = 0.0
        self.unigramDict = collections.defaultdict(int)
        self.bigramDict = collections.defaultdict(int)
        self.unigram_p = {}
        self.bigram_p = {}
        self.emission_probs = {}
        self.word_to_tags = {}
        pass

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.

        """
        tags_data = []
        for pair in data:
            tags_data.append(pair[1])
        for tag in tags_data:
            self.unigramDict[tag] += 1
        for bigram in nltk.bigrams(tags_data):
            self.bigramDict[bigram] += 1
        unigrams_len = sum(self.unigramDict.values())
        self.unigram_p = {k: math.log(float(v) / unigrams_len, 2)
                     for k, v in self.unigramDict.items()}
        self.bigram_p = {k: math.log(
            float(v) / self.unigramDict[k[0]], 2) for k, v in self.bigramDict.items()}

        unigram_p_inter = {k: math.log(
            (float(v)-1) / (unigrams_len-1), 2) for k, v in self.unigramDict.items()}
        bigram_p_inter = {k: math.log((float(v)-1) / (self.unigramDict[k[0]]-1), 2) if (float(
            v)-1) > 0 and (self.unigramDict[k[0]]-1) > 0 else 0.0 for k, v in self.bigramDict.items()}
        for k, v in self.bigramDict.items():
            self.lambda1 += v if bigram_p_inter[k] <= unigram_p_inter[k[0]] else 0.0
            self.lambda2 += v if bigram_p_inter[k] > unigram_p_inter[k[0]] else 0.0
        self.lambda1 = self.lambda1/(self.lambda1+self.lambda2)
        self.lambda2 = 1 - self.lambda1

        # emission probabilities
        self.word_to_tags = pos_tagger.wordtoTags(data)
        
        for word in self.word_to_tags:
            tags_list = self.word_to_tags[word]
            if word not in self.emission_probs:
                self.emission_probs[word] = {}
            for tag in tags_list:
                prob = math.log((tags_list[tag] / self.unigramDict[tag]), 2)
                self.emission_probs[word][tag] = prob

    def wordtoTags(self, data):
        word_to_tags = {}
        for pair in data:
            if pair[0] not in word_to_tags:
                word_to_tags[pair[0]] = {}
            if pair[1] not in word_to_tags[pair[0]]:
                word_to_tags[pair[0]][pair[1]] = 0
            word_to_tags[pair[0]][pair[1]] += 1
        return word_to_tags

    def handlingUNK(self, word):
        # Calculating UNK emission prob
        letters_to_tags ={}
        last_letters_freq = {}
        n = min(5,len(word))
        for i in range(1,n+1):
            last_letters = word[-i:]
            for key_word,tag_dict in self.word_to_tags.items():
                if key_word[-i:] == last_letters:
                    if last_letters not in last_letters_freq:
                        last_letters_freq[last_letters]=0
                    last_letters_freq[last_letters]+=sum(tag_dict.values())  
                    if last_letters not in letters_to_tags:
                        letters_to_tags[last_letters] = {}
                    for tags,freq_tag in tag_dict.items():
                        if tags not in letters_to_tags[last_letters]:
                            letters_to_tags[last_letters][tags]=0
                        letters_to_tags[last_letters][tags]+=freq_tag
        
        letters_to_tags_p = {}
            

        for last_letters, value in letters_to_tags.items():
            letters_to_tags_p[last_letters] = {}
            for tags in value:
                letters_to_tags_p[last_letters][tags] = {}
                count = letters_to_tags[last_letters][tags]
                letters_to_tags_p[last_letters][tags] = math.log(float(count)/last_letters_freq[last_letters],2)
        n = len(letters_to_tags)

        unk_prob = {}
        unk_prob[word] = {}
        std_theta = stdev(self.unigram_p.values())
        if n==0:
            unk_prob[word]["SYM"] = 0
            return unk_prob
        
        for tag in letters_to_tags_p[word[-1:]].keys():
            prob_tag_suffix = self.unigram_p[tag]
            last_letters_count = n
            for i in range(1,n+1):
                last_letters = word[-i:]
                #std_theta = stdev(letters_to_tags[last_letters].values())
                prob_tag_suffix = (letters_to_tags_p[last_letters][tag] + (std_theta*prob_tag_suffix))/(1+std_theta)
                if i!=n and letters_to_tags_p[word[-i-1:]].get(tag,0)==0:
                    last_letters_count = i
                    break
            last_letters_count = min(last_letters_count,n)
            #p(tag)
            tag_freq = 0
            for k,v in letters_to_tags.items():
                tag_freq+=v.get(tag,0)
            sum_values = sum(letters_to_tags[word[-last_letters_count:]].values())
            sum_keys = sum(last_letters_freq.values())
            tag_p = math.log(float(tag_freq)/sum_values,2)
            suffix_p = math.log(float(sum_values)/sum_keys,2)
            #Bayes Logic
            #prob_suffix_tag = math.log(float(sum(letters_to_tags[word[-last_letters_count:]].values()))/tag_freq,2)+prob_tag_suffix
            prob_suffix_tag = suffix_p-tag_p+prob_tag_suffix
            unk_prob[word][tag] = prob_suffix_tag
        return unk_prob

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        return 0.
    
    def smoothing(self, prev_tag, tag):
        return math.log((self.bigramDict[(prev_tag, tag)]+0.1)/(self.unigramDict[prev_tag]+(0.1*len(self.unigramDict))),2)


    def inference(self, sequence):
        """Tags a sequence with part of speech tags.
        """
        #print(self.bigram_p)
        emission_probs_cy = self.emission_probs.copy()
        state = [{}]
        #sequence = sequence.split(" ")

        prev_word = sequence[0]
        for tag in emission_probs_cy[prev_word].keys():
            state[0][tag] = {"p":emission_probs_cy[prev_word][tag],"prev_node":None}


        for idx in range(1,len(sequence)):
            state.append({})
            if sequence[idx] not in emission_probs_cy:
                unk_prob = pos_tagger.handlingUNK(sequence[idx])
                emission_probs_cy[sequence[idx]] = unk_prob[sequence[idx]]
            tags = list(emission_probs_cy[sequence[idx]].keys())
            prev_tags = list(emission_probs_cy[prev_word].keys())

            for tag in tags:
                transition_prob = 0.0
                
                if (prev_tags[0], tag) in self.bigram_p.keys():
                    transition_prob = self.bigram_p[(prev_tags[0], tag)]
                else:
                    transition_prob = (self.lambda1*self.unigram_p[tag]) + (self.lambda2*(self.bigram_p[(tag, prev_tags[0])] if (tag, prev_tags[0]) in self.bigram_p.keys() else pos_tagger.smoothing(prev_tags[0],tag)))
                
                
                max_tr_p = state[idx-1][prev_tags[0]]["p"] + transition_prob
                prev_state = prev_tags[0]
                for prev_tag in prev_tags[1:]:
                    transition_prob = 0.0
                    if (prev_tag, tag) in self.bigram_p.keys():
                        transition_prob = self.bigram_p[(prev_tag, tag)]
                    else:
                        transition_prob = (self.lambda1*self.unigram_p[tag]) + (self.lambda2*(self.bigram_p[(tag, prev_tag)] if (tag, prev_tag) in self.bigram_p.keys() else pos_tagger.smoothing(prev_tag,tag)))
                    
                    tr_p = transition_prob + state[idx-1][prev_tag]["p"]
                    if(tr_p>max_tr_p):
                        max_tr_p = tr_p
                        prev_state = prev_tag
                v_p = max_tr_p + emission_probs_cy[sequence[idx]][tag]
                state[idx][tag] = {"p":v_p,"prev_node":prev_state}
            prev_word = sequence[idx]
        
        result = []
        max_prob = float("-inf")
        best_st = None

        for tag, prob in state[-1].items():
            if prob["p"]>max_prob:
                max_prob = prob["p"]
                best_st = tag
        result.append(best_st)
        previous_state = best_st

        for idx in range(len(state)-2,-1,-1):
            result.insert(0, state[idx+1][previous_state]["prev_node"])
            previous_state = state[idx+1][previous_state]["prev_node"]
        return result

    


if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv","data/dev_y.csv")
    '''
    dev_words = []
    dev_tags = []
    for pair in dev_data:
        dev_words.append(pair[0])
        dev_tags.append(pair[1])
    #test_data = load_data("data/test_x.csv")

    test_data_formatted = []
    sentence_data = ""
    for data in dev_words:
        if data == '-docstart-':
            test_data_formatted.append(sentence_data)
            sentence_data = ""
        sentence_data += data + " "
        pass

    pos_tagger.train(train_data)

    # evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    test_data_tags=[]
    for sentence in test_data_formatted[1:]:
        test_data_tags.extend(pos_tagger.inference(
            sentence))
    # Write them to a file to update the leaderboard
    '''
    
    
    
    
    dev_words = []
    dev_tags = []
    for pair in dev_data:
        dev_words.append(pair[0])
        dev_tags.append(pair[1])
    pos_tagger.train(train_data)

    '''
    dev_sim_tags = pos_tagger.inference(dev_words)
    print(len(dev_sim_tags))
    print(len(dev_tags))
    check = [i for i, j in zip(dev_tags, dev_sim_tags) if i == j] 
    accuracy = len(check)/len(dev_tags)
    print(accuracy*100)
    evaluate(dev_data, pos_tagger)
    '''
    

    
    '''
    dev_sim_tags_dict = []
    idx=0
    for tag in dev_sim_tags:
        dev_sim_tag_dict = {}
        dev_sim_tag_dict["id"] = idx
        dev_sim_tag_dict["tag"] = '\"'+tag + '\"'
        dev_sim_tags_dict.append(dev_sim_tag_dict)
        idx+=1
    csv_header = ["id","tag"]
    with open("pred_y.csv", 'w') as file:
        writer = csv.DictWriter(file, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(dev_sim_tags_dict)
    '''

    test_data = load_data("data/test_x.csv")
    test_sim_tags = pos_tagger.inference(test_data)
    test_sim_tags_dict = []
    idx=0
    for tag in test_sim_tags:
        test_sim_tag_dict = {}
        test_sim_tag_dict["id"] = idx
        test_sim_tag_dict["tag"] = '\"'+tag + '\"'
        test_sim_tags_dict.append(test_sim_tag_dict)
        idx+=1
    csv_header = ["id","tag"]
    with open("results_hmm/test_y.csv", 'w') as file:
        writer = csv.DictWriter(file, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(test_sim_tags_dict)
    

    # TODO
