import sys
import nltk
from nltk import word_tokenize,sent_tokenize
import re
import keras
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, RepeatVector,Merge
from keras.layers import LSTM
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
from keras.models import model_from_json
import h5py
import numpy as np
from keras.models import load_model
import itertools
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import gensim
from gensim.models import word2vec
import imblearn

from imblearn.over_sampling import SMOTE


corpus=[]
def tokenize(sent):
	sent=sent_tokenize(sent)
	#print(sent)
	return(sent)

def Extracting_Data(lines):
	data = []
	for i,line in enumerate(lines):
		#print(line,i)
		#print("\n")
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
		#print("information: ",information)
		#print("question: ",question)
		#print("\n")
		corpus.append([information,question])
		data.append((information,question,answer))
		
	return data


def get_Data(f):
	data = Extracting_Data(f.readlines())
	return data




train = get_Data(open('DATA/train_New.txt', 'r'))
test = get_Data(open("DATA/test_New.txt", 'r'))


print("\n---------------------------\n")
print(train[:10],len(train))
print("\n----------------------------\n")

print("corpus: ",corpus[:10],len(corpus))


print("\n")


import itertools

corpus=list(itertools.chain(*corpus))

model=word2vec.Word2Vec(corpus,min_count=1,size=22)

# print vector of that word
print model["soccer"]
print("\n\n")

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



vocab = sorted(reduce(lambda x, y: x | y,(set(story + q + [answer]) for story, q, answer in train + test)))

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
#print('a', word_idx_answer,word_idx_operator_reverse)
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train+ test)))
query_maxlen=story_maxlen # TAKING SAME DIMENSION FOR STORY AND QUESTION....




def get_word2vec(data, word_idx, word_idx_answer, story_maxlen, query_maxlen):
	X = []
	Xq = []
	Y = []
	print("length of data")
	print(len(data))
	for story, query, answer in data:
		x = [model[w] for w in story]
		xq = [model[w] for w in query]

		y = np.zeros(len(word_idx_answer))
		for item in answer.split():
			if re.search('\+|\-|\*|/', item):
				y[word_idx_answer[item]] = 1


		x=list(itertools.chain(*x))
		xq=list(itertools.chain(*xq))
		X.append(x)
		Xq.append(xq)
		Y.append(y)

	return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)




################## Word2Vector Representation ##############################

'''
X, Xq, Y = get_word2vec(train, word_idx, word_idx_answer, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_Data(test, word_idx, word_idx_answer, story_maxlen, query_maxlen)
expected_T=[np.argmax(le) for le in Y] #train class labels


print("\n------------------------------------------\n")
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('tX.shape = {}'.format(tX.shape))
print('tXq.shape = {}'.format(tXq.shape))
print('tY.shape = {}'.format(tY.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))
print("\n---------------------------------------------\n")

'''

############################# word Index Vector Representation ####################
X, Xq, Y = vectorize_Data(train, word_idx, word_idx_answer, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_Data(test, word_idx, word_idx_answer, story_maxlen, query_maxlen)
expected_T=[[np.argmax(le)] for le in Y] #train class labels


print("\n------------------------------------------\n")
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('tX.shape = {}'.format(tX.shape))
print('tXq.shape = {}'.format(tXq.shape))
print('tY.shape = {}'.format(tY.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))
print("\n\n")




print("\n--------------------- Schema Identification ---------------------------\n")
print('Build model ...')
print("\n")
print("MultiLayer Perceptron Classifier ...\n-----------------------------\n\n")
X_new=X+Xq
tX_new=tX+tXq


print("Before OVer Sampling train data : ",len(X_new),X_new.shape)
print("y: ",len(expected_T),expected_T[:10])
print("\n")


sm = SMOTE(random_state=1)
X_new, expected_T = sm.fit_sample(X_new, np.array(expected_T).ravel())

#print("After OVer Sampling Train data : ",X_new.shape,expected_T.shape,expected_T[:10])
print("\n\n")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(350, 220,50), random_state=1,activation='relu') ##relu,lbfgs,adam is okay....


#clf.fit(X_new, expected_T)
clf.fit(X_new, expected_T) ## Oversampling fit
pred=clf.predict(tX_new)
pred_T=clf.predict(X_new)


print("Training Samples: ",len(X_new))
print("Testing Samples: ", len(tX_new))
print("\n\n")

#target_names=["+","-","/","*"]
target_names=["+","*","-","/"]


expected=[np.argmax(le) for le in tY]
#predicted=[np.argmax(le) for le in pred] #we dont require one-hot vector
predicted=pred



print("\n\n---------------Train Data Accuracy Matrix-------------------\n")
print(classification_report(expected_T, pred_T, target_names=target_names))
print("\n----------------------------------\n")

print("\n\n---------------Test Data Accuracy Matrix-------------------\n")
print(classification_report(expected, predicted, target_names=target_names))
print("\n----------------------------------\n")
print("Train Accuracy: ",accuracy_score(expected_T, pred_T)*100)
print("\n---------------------------------\n")
print("Test Accuracy: ",accuracy_score(expected, predicted)*100)
print("\n---------------------------------\n")
print("\n")
print(clf)



print("\n********************** Equation & Answer Generation **************************\n")

information=[['Joan', 'found', '5', 'seashells', 'on', 'the', 'beach', '.'],[ 'she', 'gave', 'Sam', 'some', 'of', 'her', 'seashells', '.'], ['She', 'has', '1', 'seashells', '.']]

questions=[]

for sample in test:
	print(sample)
	#information=sample[0]
	questions=sample[1]
	break


'''
question_tokens=word_tokenize(questions)

tokens=[]
tokens.append([word_tokenize(information[i]) for i in range(len(information))])

'''

question_tokens=questions

tokens=information
print(tokens,len(tokens))
print("\n\n")

sent_own_obj_dict={}
for i,line_tokens in enumerate(tokens):
	print(line_tokens,i)
	pos_tags_line=nltk.pos_tag(line_tokens)
	print("pos_tags_line",pos_tags_line)
	owner_tags=['NNP','NNS','NN','NNPS']
	verb_tags=['VBD','VBZ','VBP','VBN','VBG','VB']
	object_tags=['NNS','NNPS']

	owner=[]
	verbs=[]
	objects=[]
	adj_word=''
	temp_val=''

	for j,w in enumerate(pos_tags_line):
		if(w[1] in owner_tags):
			owner.append(w)

		if(w[1] in object_tags):
			print("obj.. ",w[1],object_tags)
			if((pos_tags_line[j-1][1])=='JJ'):
				adj_word=pos_tags_line[j-1][0]
				word=adj_word+"-"+w[0]
				print("word:" ,word,w) # ALert
				w_mod=(word,w[1])
				objects.append(w_mod)

			else:
				print("else obj tags")
				objects.append(w)
				
		if(w[1] in verb_tags):
			verbs.append(w)
		if(w[1]=='CD'):
			entity_value=w
			temp_val=w[0]


	if not (temp_val):
		# dummy sentences....no use of that sentence
		#del information[i]
		#i=i-1
		temp_val=99999 #dummy entity value
		#continue




	k1="owner"+str(i)
	k2="object"+str(i)
	k3="entity_value"+str(i)
	k4="verb"+str(i)

	v1=owner[0][0]
	v2=objects[0][0]
	v4=verbs[0][0]


	temp={}
	temp[k1]=v1
	temp[k2]=v2
	temp[k3]=temp_val
	temp[k4]=v4
	sent_own_obj_dict[i]=temp
	 
		




print("\n\n")
print("sent_own_obj_dict: ",sent_own_obj_dict)
print("\n\n")


#exit()

question_tokens_tags=nltk.pos_tag(question_tokens)
print(question_tokens_tags)

pronoun_tag=['PRP','NN','NNP']
verb_tag=['VBP','VBZ','VBD','VB']
object_tag=['NNS','NNPS']

print("\n\n")
print(sent_own_obj_dict)
print("\n")

not_adj=['much','many','more']
adj_word=''
start=''
last=''
# If more than number of possiblities occurs then it will take last one....IN case of Verbs
for i,w in enumerate(question_tokens_tags):
	if(w[1] in pronoun_tag):
		start=w[0]
	if(w[1] in verb_tag):
		second=w[0]
		
	if(((question_tokens_tags[i-1][1])=='JJ') ):
		adj_word=question_tokens_tags[i-1][0]
		if(adj_word in not_adj):
			last=w[0]
			print("If last not adj",last,w)
		else:
			last=adj_word+"-"+w[0]
			print("If last",last,w)



if not(start):
	start="They"
if not(last):
	last=sent_own_obj_dict[0]["object0"]

print("=============> ",start,second,last)


print(pred)
print("\n\n")


print()


def Combine(sent_own_obj_dict):
	values=[]
	for i in range(len(information)):
		x="entity_value"+str(i)
		#print("x",x)
		xx="object"+str(i)

		if(last==sent_own_obj_dict[i][xx]):
			values.append(int(sent_own_obj_dict[i][x]))	
	print(values)
	values. remove(99999)
	print(values)
	result=sum((values))
	print("answer: ",result)
	third=result
	print("Answer in Natural Language Processing....")
	print(start+" "+second+" "+str(third)+" "+last)

Combine(sent_own_obj_dict)




def addition(numbers):
	ans=0
	for num in numbers:
		ans+=num
	return ans


def multiplication(numbers):
	ans=1
	for num in numbers:
		ans*=num
	return ans



def substraction(numbers):

	pass


def division(numbers):
	
	pass
	

