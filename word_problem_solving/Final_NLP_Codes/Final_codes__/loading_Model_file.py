from __future__ import print_function
from functools import reduce
import re
import sys
import nltk
from nltk import word_tokenize,sent_tokenize
import re
import keras

#from keras.layers import merge
#from keras.engine import merge
#from keras.layers import Dense, Merge
#from keras.layers import add

from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, RepeatVector,Merge


from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
from keras.models import model_from_json
import h5py
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import load_model
import itertools



def tokenize(sent):
	sent=sent_tokenize(sent)
	print(sent)
	return(sent)

def Extracting_Data(lines):
	data = []
	for line in lines:
		information=[]
		question=[]
		line=line.strip().split("\t")
		sent=sent_tokenize(line[0])
		answer=line[1]
		for i,sn in enumerate(sent):
			if(re.search('\?' , sn)):
				question.append(word_tokenize(sn))
			else:
				information.append(word_tokenize(sn))
		information=list(itertools.chain(*information))
		question=list(itertools.chain(*question))
		data.append((information,question,answer))		
	return data


def get_Data(f):
	data = Extracting_Data(f.readlines())
	return data

train = get_Data(open('/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/DATA/train_New.txt', 'r'))
test = get_Data(open("DATA/test2.txt", 'r'))
print("train: ",train)
print("\n")


def vectorize_Data(data, word_idx, word_idx_answer, story_maxlen, query_maxlen):
	X = []
	Xq = []
	Y = []
	print("length of data")
	print(len(data))
	for story, query, answer in data:
		x = [word_idx[w] for w in story]
		xq = [word_idx[w] for w in query]
		y = np.zeros(len(word_idx_answer))
		for item in answer.split():
			if re.search('\+|\-|\*|/', item):
				y[word_idx_answer[item]] = 1
		X.append(x)
		Xq.append(xq)
		Y.append(y)

	return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)



RNN = recurrent.GRU
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))

vocab = sorted(reduce(lambda x, y: x | y,
                      (set(story + q + [answer]) for story, q, answer in train + test)))

vocab_size = len(vocab) + 1
vocab_answer_set = set()
for story, q, answer in train + test:
	for item in answer.split():
		if re.search('\+|\-|\*|/', item):
			vocab_answer_set.add(item)

vocab_answer = list(vocab_answer_set)
vocab_answer_size = len(vocab_answer)

word_idx = OrderedDict((c, i + 1) for i, c in enumerate(vocab))
word_idx_answer = OrderedDict((c, i) for i, c in enumerate(vocab_answer))
word_idx_operator_reverse = OrderedDict((i, c) for i, c in enumerate(vocab_answer))
print('a', word_idx_answer,word_idx_operator_reverse)
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train+ test)))
print("query_maxlen", query_maxlen , story_maxlen)

X, Xq, Y = vectorize_Data(train, word_idx, word_idx_answer, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_Data(test, word_idx, word_idx_answer, story_maxlen, query_maxlen)

print('X.shape = {}'.format(X.shape), X)
print('Xq.shape = {}'.format(Xq.shape), Xq)
print('Y.shape = {}'.format(Y.shape), Y)
print('tX.shape = {}'.format(tX.shape))
print('tXq.shape = {}'.format(tXq.shape))
print("\n")
print('tY.shape = {}'.format(tY.shape),tY)
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))





target_names=["+","*","-","/"]
################################## Test Input #######################3
load_model = load_model('my_model.h5')
loss, acc = load_model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE)
print("Training evaluation..................")
print(' loss - accuracy = {:.4f} / {:.4f}'.format(loss, acc))
print("\n\n")
print("Testing..............")
print('loss -accuracy = {:.4f} / {:.4f}'.format(loss, acc))
loss, acc = load_model.evaluate([X, Xq], Y, batch_size=BATCH_SIZE)
print("\n\n")
print("Test Sample: \n")
predicted_Labels=load_model.predict([tX, tXq])
print(predicted_Labels)
target_names=["+","*","-","/"]
print(predicted_Labels)
labels_test=[np.argmax(lb) for lb in predicted_Labels]
print("labels_test: ",labels_test)
