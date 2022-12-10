1        Introduction


This project aims to build a model to identify the Part of Speech of each word in the data. The goal is to attain more than 90% accuracy for finding the POS tag of an unknown word. The algorithm Hidden Markov Model was implemented to calculate the probabilities and Viterbi is implemented to estimate the most probable tag for a word. We refer to the 48 tags from the Penn Treebank corresponding to 36 words and 12 special characters. We built our model by using data from POS tagged sentences as training data. We have handled unknown words in the model implementing smoothing techniques and increased the accuracy. 
  



2        Data


We have a train dataset that contains almost a million words of text from the Wall Street Journal. The sentences in the dataset are written out word-by-word in a flat, column format in the csv files. Each of these words has a corresponding POS tag in another csv file. We have a train dataset with words file and tags file which we used to train the model. We preprocessed the data by removing the quotes and converting all words to lowercase. In the train dataset, we clubbed the words to respective tags as a pair for ease of implementation.And a development dataset of words file which is used to predict the tags and a tags file to evaluate the model. Now, we have a test dataset of only words file for which we calculate the POS tags file, and write it into a csv file. 
Dataset
	Count
	Train dataset
	696474
	Development dataset
	243020
	Test dataset
	236581
	Figure: Datasets and their size.


3        Implementation


The first thing is to decide on the implementation logic and how many methods and data structures we would like to have. We have used lists and dictionaries at different places as per the requirement of the logic. Below is an image of the data structures used.


  

Figure 1: Data structures used code snippet.


We implemented HMM for calculating transition and emission probabilities. Transition probability is the probability of a tag given the prev tag. And emission probability is the probability of a word to be a given tag. For transition probabilities we considered the tags and used Bigram Language models to know the probability of occurrence of a tag after another tag. For this purpose, we need to know the bigrams from all the tags and we took help of the bigrams() method from the NLTK package to get the possible bigrams and then we calculated the respective maximum likelihood probabilities. 


Bigram probabilities generated from the transition states of the training data cannot directly be used in the model because of sparse data problems. To this problem, we have solutions of smoothing like Laplace smoothing, Add-K smoothing, Linear Interpolation. For this model we used Linear Interpolation. So we estimate a bigram probability as follows:


  



  
 is the maximum likelihood estimates of the probabilities and         


The values of weights are calculated by deleted interpolation and the algorithm is as follows:


  



By implementing the above algorithm we found lambda1 = 0.31724802361610055, lambda2 = ​​0.6827519763838994


  

Figure: Linear Interpolation code snippet.


Next, we collected the counts of different tags for each word into a dictionary of words and its respective dictionary of tags and frequencies. This dictionary was further used to calculate the emission probabilities. For this we used the frequencies of unigram tags to know the no of occurrences of a tag. In this case, we only calculate the emission probabilities of words in train data. With the help of these and using suffix algorithms, we handle the unknown words and calculate the emission probabilities for the same. 


Handling of Unknown Words:


In this model, we used Suffix analysis to handle unknown words. The probability distribution for a particular suffix is generated from all words in the training set that share the same suffix of some predefined maximum length. The term suffix as used here means “final sequence of characters of a word” which is not necessarily a linguistically meaningful suffix. Probabilities are smoothed by successive abstraction. This calculates the probability of a tag t given the last m letters l of an n letter word is   . We used probabilities of consecutive combinations suffixes of m letters for the calculation. 


So the recursive formula is follows :


  

Where i = m,m-1,...0
  
 is the maximum likelihood estimates of the probabilities and the initialization   
The maximum likelihood estimates of the probabilities of length i derived from the following formula:
  



For the Markov model, we need inverse conditional probabilities    , we can be achieved by Bayesian inversion formula. And the value of θ is the standard deviation of the maximum likelihood probabilities of all tags. We used the stdev method from the statistics package to calculate the value of θ.


  

Figure: Handling unknown words code snippet.


Finally, we are ready to start working on the Viterbi algorithm. Viterbi algorithm is also a dynamic programming approach. In this algorithm, we reused the calculated result in the next calculation. It is a natural choice for HMM, in which any state depends on its first previous state. The effects of the other previous states are all absorbed into the first previous state, which is now the only factor you need to take into account when transitioning to the next state.The Viterbi algorithm's goal is to draw a conclusion from some observed data and a trained model.


To calculate the most probable state given the training model and data, we multiply the transition probability of the current state and previous tag with the probability of previous state and tag. Once we get the highest probability for a tag considering all the previous tags. We save the probability of a word at this state along with its probability and the previous state. We calculate this way for the words and possible tags. In this process we do Add-k smoothing for unknown (prev_tag, tag) pairs in the transition matrix. 


  

Figure: Viterbi Logic code snippet.


By the time we are done calculating the probabilities for all the words in the data, we will have different probabilities with traceable paths. We consider the path that results in the maximum probability and trace back in that path with the help of the previous state we saved in the state matrix. In this way we identify the tags for the respective words.


  

Figure: Backtracking the max probability path code snippet.


Once we get the list of predicted tags for the words, we write that to a csv file with the help of csv package. For development data, we compare the predicted tags file and actual tags file to calculate the accuracy of our model. 


The two design choices that helped us increase the efficiency of the model are adding lambda weights to the transition matrix with the help of unigram probabilities. The thought process is to solve the sparsity problem in the input data. Handling the unknown probabilities will help in better predicting the tags of the words. Another choice is to add epsilon values to the emission matrix such that all the tag probabilities are calculated for a word. For the scenarios in which we do not have data, we keep it a small value epsilon in order for that scenario to be considered and better predict the tags. The normalization helped in improving the accuracy by 4%.


4        Experimental Design and Evaluations


Considering the collection of unknown words in a sentence and predicting the part of speech tag is a significant challenge. We simplified this difficulty by using the sophisticated literature summarization technique here called Hidden Markov Models (HMM). Despite considering well known taggers like TnT POS tagger, we took up this work using Bigram probability estimates as the last level of transition probability. We assumed that the suffix of a word is a strong predictor of POS Tagger and we expected to be able to calculate the accurate probabilities and identify the relevant tag.Linear interpolation calculations supported the accuracy with smoothing the transition probabilities to make sure all transitions are considered in the part of Vertibi calculations


Evaluation of the model is done by using development data and we used the script provided and calculated F1 score of HMM POS tagger model. The score of the model is Mean F1 Score: 0.9195807402714128. And we uploaded test data, predicted tags and evaluated on the Leaderboard and its F1 score is ____.


5        Results


Dataset
	Accuracy
	F1 score
	Train
	95.98377544061165
	0.9687465917359878
	Development
	91.99904534999033
	0.9195807402714128
	Figure: Performance of system for various datasets.


  

Figure: Leaderboard score


6        Analysis


The accuracy of our model is around 92%. Even though the training data is good enough to make the model fit, we believed that we needed to handle the words with lower frequency values. To handle the lower frequency words we might need to consider smoothing like Add-K smoothing. Moreover, the last level of language model we used here is Bigram. Referring to other sources of information, we learned that we can get good accuracy with Trigrams, Linear interpolation smoothing  and handling unknown words with suffix analysis.


We categorized the error analysis by segregating the words into two parts, one is word containing only special characters and other is word containing only alphanumeric characters.


The accuracy of alphanumeric words prediction is 94.44477226247938.
The accuracy of special characters words prediction is 85.93032700877703.


Another way of error analysis we observed by segregating the words into Known words and Unknown words.
The accuracy of Known words prediction is 96.54417422161877.
The accuracy of Unknown words prediction is 87.95807402714128.


7        Conclusion


The experiment aimed to build a model to identify the POS tag for given data. The Viterbi algorithm used is decently efficient along with the implementation of unknown word handling, linear interpolation and add k smoothing. Each algorithm helped to calculate the probabilities more accurately and eventually identify the appropriate tag.


The major learning curve is implementation of the Viterbi algorithm as it is quite challenging to understand the flow and keep track of states. Understanding the data was one of the crucial steps before we started through the thought process for the solution. Next, figuring out the right algorithms also helps in understanding how each algorithm affects the calculations of the probability in the viterbi.


Other


We brainstormed through multiple discussions to understand the data and algorithms we need to explore for the project. We coordinated for implementation of the logic and for summarizing the report.


Feedback


The given assignment helped us understand the POS tagging and its implementation thoroughly. After working with HMM and Viterbi algorithms, the concept behind how tags are predicted is more clear. Consistently evaluating the model and smoothing it to improve the accuracy helped to learn more about new algorithms. Although we faced some challenges with low probabilities, we handled it using logarithmic implementation. The project was decently difficult and had a huge learning curve. We referred to a couple of research papers for understanding certain challenging concepts.
