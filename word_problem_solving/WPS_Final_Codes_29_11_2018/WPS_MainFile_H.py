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
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import gensim
from gensim.models import word2vec
import sys, os
os.system
import subprocess
from subprocess import Popen, PIPE
import re


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




train = get_Data(open('/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/stanford-parser-full-2018-10-17/DATA/train_New.txt', 'r'))
test = get_Data(open("/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/stanford-parser-full-2018-10-17/DATA/test_New.txt", 'r'))


import itertools

corpus=list(itertools.chain(*corpus))

model=word2vec.Word2Vec(corpus,min_count=1,size=100)

def vectorize_Data(data, word_idx, word_idx_answer, story_maxlen, query_maxlen):
	X = []
	Xq = []
	Y = []
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




################# Loading Dataset ###################
train = get_Data(open('/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/stanford-parser-full-2018-10-17/DATA/train_New.txt', 'r'))
test = get_Data(open("/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/stanford-parser-full-2018-10-17/DATA/test_New.txt", 'r'))
#test = get_Data(open("DATA/test2.txt", 'r'))


######################## Vocab & word index Formation ##############
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
############################## word Index Vector Representation #########################
X, Xq, Y = vectorize_Data(train, word_idx, word_idx_answer, story_maxlen, query_maxlen)
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
print("\n")





############################ Schema Identification ###########################
print("\n--------------------- Schema Identification ---------------------------\n")
print('Build model ...')
print("\n")
print("MultiLayer Perceptron Classifier ...\n\n")
X_new=X+Xq
tX_new=tX+tXq

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(350, 220,50), random_state=1)
clf.fit(X_new, expected_T)
pred=clf.predict(tX_new)
pred_T=clf.predict(X_new)

#print("expected_labels: ",expected_T)
print("Predicted labels: ",pred)
print("\n")

target_names=["+","*","-","/"]

expected=[np.argmax(le) for le in tY]
#predicted=[np.argmax(le) for le in pred] #we dont require one-hot vector
predicted=pred

print("\n Train matrix : \n----------------------------------\n")
print(classification_report(expected_T, pred_T, target_names=target_names))
print("\n Test Matrix : \n----------------------------------\n")
print(classification_report(expected, predicted, target_names=target_names))
print("\n----------------------------------\n")
print("MLP Classifier Train Accuracy: ",accuracy_score(expected_T, pred_T)*100)
print("\n---------------------------------\n")
print("MLP Classifier Test Accuracy: ",accuracy_score(expected, predicted)*100)
print("\n---------------------------------\n")
print("\n")





############################### Seq2Seq Model Classifier ###########################

print("\n ################## Seq2Seq Model Classifier  ################\n")
from loading_Model_file import labels_test
print("Predicted_Test LAbel: ",labels_test)
print("****************")
expected=[np.argmax(le) for le in tY]
print("\n---------------------------------\n")
print("seq2seq Classifier Test Accuracy: ",accuracy_score(expected, labels_test)*100)
print("\n------------------------------------------------------------------\n")
print("\n\n")



########################### SMOTE: OVerSampling ##########################
print("################## SMOTE: OVerSampling ##############\n")
print("Before OVer Sampling train data : ",len(X_new),X_new.shape)

sm = SMOTE(random_state=9)
X_new, expected_T = sm.fit_sample(X_new, np.array(expected_T).ravel())

clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(350, 220,50), random_state=1)##relu,lbfgs,adam
clf.fit(X_new, expected_T)
pred=clf.predict(tX_new)
pred_T=clf.predict(X_new)

target_names=["+","*","-","/"]

expected=[np.argmax(le) for le in tY]
#predicted=[np.argmax(le) for le in pred] #we dont require one-hot vector
predicted=pred


print("\n Train matrix : \n----------------------------------\n")
print(classification_report(expected_T, pred_T, target_names=target_names))
print("\n Test Matrix : \n----------------------------------\n")
print(classification_report(expected, predicted, target_names=target_names))
print("\n----------------------------------\n")
print("MLP Classifier Train Accuracy: ",accuracy_score(expected_T, pred_T)*100)
print("\n---------------------------------\n")
print("MLP Classifier Test Accuracy: ",accuracy_score(expected, predicted)*100)
print("\n---------------------------------\n")
print("\n")



####################################### Modules for Pre-Processing ################################


print("\n********************** Equation- Answer Generation **************************\n\n\n")


#####Module1 Resolve_Currency#############
def Resolve_Currency(inf):
	print ("Sent received at Currency module is:\n",inf)
	word_token = inf.split(" ")
	mod_sentence=""
	for token_curr in word_token:
		if(re.search(r"\$",token_curr)):
			mod_sentence=mod_sentence+" "+token_curr[1:]+" "+"dollars"
		else:
			mod_sentence=mod_sentence+" "+token_curr
	return(mod_sentence[1:])

def Resolve_Conjunctions(inf,i):
	###Applying POS tagging before splitting the sentence#####
	word_token = inf.split(" ")
	word_token = [X for X in word_token if X!=""]
	#print("Word Tokens ", word_token)
	pos_tags_sent=[]
	temp=nltk.pos_tag(word_token)
	pos_tags_sent.append(temp)
	#print("\n\n")
	#print(pos_tags_sent)
	#print("\n\n")
	resolve_pos_dict=dict(pos_tags_sent[0])
	CC_resolve_pos_dict[i]=resolve_pos_dict
	CC_resolve_pos_dict[i+1]=resolve_pos_dict
	new_sent=re.split(r"(?:\band\b|\bif\b|\bbut\b|\bso\b)", inf)
	#print("CC Resolve sent is ", new_sent)
	print("\n")
	mod_sentence=''
	for sent in new_sent:
		mod_sentence+=sent+'.'
	return mod_sentence[:-2]

Owner_Entity_dict={}

def Extract_Roles_entities(processed_information,CC_resolve_pos_dict,conjunction_index):

	dict_cc_resolve={}

	####Word Tokenizing the information sentences####
	tokens=[word_tokenize(new_sent_sample) for new_sent_sample in processed_information]
	"""print("\n----------------tokens -------------------\n")
	print("tokens:",tokens)
	print("\n\n-------------- POS tags ----------------\n")"""


	#iterate over the all sentences:
	#verb=[]
	Entity_Values=[]
	irrelevent=0
	FlagMod=0
	for i,sent in enumerate(tokens):
		temp={}
		verb=[]
		mod_I=i
		
		for j,w in enumerate(sent):
			Entities=[]
			if(CC_resolve_pos_dict[i][w] in verb_tags):
				Owners1=[]
				Owners2=[]
				#print("IN IF")
				verb=[w]
				if(j!=0):
					pre_verb=sent[:j]
					pre_verb_tags = [CC_resolve_pos_dict[i][w] for w in sent[:j]]
					#print("PREverb is" , pre_verb)
					for o,tag in enumerate(pre_verb_tags):
						#print("PREverb is" , pre_verb, tag)
						if (tag in Noun_tags or tag == "EX"):
							#print("In Noun tags ",sent[:j],sent[o])
							Owners1.append(sent[o])
							#print("Owners1 is ", Owners1)
						if (tag in pronoun_tags):
							#print("In Pronoun Tag")
							if("Owners1" in dict_cc_resolve[mod_I-1].keys() and mod_I!=0):
								Curr_Owners=dict_cc_resolve[mod_I-1]["Owners1"]
								#print("***************",mod_I, Curr_Owners) 
								pre_verb=dict_cc_resolve[mod_I-1]["pre_verb"]
								Owners1.append(Curr_Owners[0])
								#print("Owners1 2nd Prp is ", Owners1)
						elif (Owners1==[] and mod_I!=0):
							if("Owners1" in dict_cc_resolve[mod_I-1].keys()):
								Curr_Owners=dict_cc_resolve[mod_I-1]["Owners1"]
								Owners1.append(Curr_Owners[0])
							
					
					#if (pre_verb in 
				else:
					#print("In else")
					if(mod_I!=0):
						if("Owners1" in dict_cc_resolve[mod_I-1].keys()):
							pre_verb = dict_cc_resolve[mod_I-1]["pre_verb"]
							Owners1 = dict_cc_resolve[mod_I-1]["Owners1"]
					else:
						pre_verb=None
				after_verb_tags=[CC_resolve_pos_dict[i][w] for w in sent[j+1:]]
				Averb=[]
				prp_verb=[]
				
				Prp_verb_Tags=[]
				#####Storing the entity vlues:
				for k,tag in enumerate(after_verb_tags):
					#print("Tag is ", tag)
					if (tag in object_tag):
						Entities.append(sent[j+1+k])
					if (tag == "NNP"):
						#print("In Owner2 tags", sent[j+k+1])
						Owners2.append(sent[j+k+1])
					if(tag =="CD"):
						#temp[Entity_value_Tag] = sent[j+k+1]
						Entity_Values.append(sent[j+k+1])
					if not (tag in Prep_tags):
						Averb.append(sent[j+1+k])
					else:
						prp_verb=sent[j+1+k:]
						Prp_verb_Tags=[CC_resolve_pos_dict[i][w] for w in sent[j+1+k:]]
						break
				for o2,tag in enumerate(Prp_verb_Tags):
					if (tag in object_tag):
						Entities.append(prp_verb[o2])
					if(tag =="CD"):
						Entity_Values.append(prp_verb[o2])
					if (tag == "NNP"):
						Owners2.append(prp_verb[o2])
				if (Entities==[] and mod_I!=0):
					if("Entities" in dict_cc_resolve[mod_I-1].keys()):
						Entities = dict_cc_resolve[mod_I-1]["Entities"]
				
					

				after_verb=Averb

			else:
				if(j==len(sent)-1 and mod_I==0 and verb==[]):
					#print ("First Sentence , no verb", sent)
					pre_verb = sent
				if(j==len(sent)-1 and mod_I!=0 and verb==[]):
					#print("###########Verb is not in the sentence#############: \n")
					#print("prev_sent,Current_sent : ", i , tokens[i-1],sent)
					#print("\n")
					#print("Dict is ", dict_cc_resolve[i-1], j)
					if("pre_verb" in dict_cc_resolve[mod_I-1].keys()):
						pre_verb = dict_cc_resolve[mod_I-1]["pre_verb"]
					#print("**********Owners1", pre_verb)
					if("Owners1" in dict_cc_resolve[mod_I-1].keys()):
						Owners1= dict_cc_resolve[mod_I-1]["Owners1"]
					if("verb" in dict_cc_resolve[mod_I-1].keys()):
						verb= dict_cc_resolve[mod_I-1]["verb"]
					after_verb_tags=[CC_resolve_pos_dict[i][w] for w in sent]
					Averb=[]
					prp_verb=[]
					Prp_verb_Tags=[]
					for k,tag in enumerate(after_verb_tags):
						if (tag in object_tag):
							Entities.append(sent[k])
						if(tag =="CD"):
							Entity_Values.append(sent[k])
						if not (tag in Prep_tags):
							Averb.append(sent[k])
						else:
							prp_verb=sent[k:]
							Prp_verb_Tags=[CC_resolve_pos_dict[i][w] for w in sent[k:]]
							break
					for o2,tag in enumerate(Prp_verb_Tags):
						if (tag in object_tag):
							Entities.append(prp_verb[o2])
						if(tag =="CD"):
							Entity_Values.append(prp_verb[o2])
						if (tag == "NNP"):
							Owners2.append(prp_verb[o2])
					if (Entities==[]):
						if("Entities" in dict_cc_resolve[mod_I-1].keys()):
							Entities = dict_cc_resolve[i-1]["Entities"]				
 
					after_verb=Averb

		temp["verb"]=verb
		temp["pre_verb"]=pre_verb
		if("after_verb" in temp.keys()):
			temp["after_verb"]=after_verb
		if("prp_verb" in temp.keys()):
			temp["prp_verb"]=prp_verb
		if("Entities" in temp.keys()):
			temp["Entities"]=Entities
		if("Owners1" in temp.keys()):
			temp["Owners1"]=Owners1
		if("Owners2" in temp.keys()):
			temp["Owners2"]=Owners2
		if("Entity_Values" in temp.keys()):
			temp["Entity_Values"]=Entity_Values

		dict_cc_resolve[mod_I]=temp
	
		i=mod_I
	return(dict_cc_resolve)








def addition(numbers):
	print("Schema Identified :", "  Addition ")
	result=sum((numbers))
	ans=result
	return ans


def multiplication(numbers):
	print("schema Identified: "," Multiplication ...")
	ans=1
	for num in numbers:
		ans*=num
	print("answer: ",ans)
	return ans

def substraction(numbers):
	
	print("schema Identified: "," Substraction")
	ans=abs(numbers[0]-numbers[1])	
	print("answer: ",ans)
	return ans


def division(numbers):
	print("schema Identified: "," Division")
	if(numbers[1]>numbers[0]):
		ans=numbers[1]/numbers[0]
	else:
		ans=numbers[0]/numbers[1]
	print("answer: ",ans)
	return ans


def Generate_Answer(i, values):
	print("Test sample, Predicted_class: ",i+1,pred[i])
	if(pred[i]==0):
		ans=addition(values)
	if(pred[i]==2):
		ans=substraction(values)
	if(pred[i]==3):
		ans=division(values)
	if(pred[i]==1):
		ans=multiplication(values)
	return ans
	print("\n----------------------------------------\n")







##########Processing the Question#################

test = open("/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/stanford-parser-full-2018-10-17/DATA/test_New_wt_Values.txt", 'r')
#test = open("DATA/test.txt", 'r')

conjunction_keywords=["and","but","if","then"]

currency_keywords=["$"]

verb_tags=['VBD','VBZ','VBP','VBN','VBG','VB']
Noun_tags=['NNP','NN',"NNS"]
Prep_tags=['IN','TO']
object_tag=['NNS','NNPS']
pronoun_tags=['PRP','PRP$']
conjunction_index=99 # number of sentences not exceeding 99 bcz of CC resolve

for test_sample_index,sample in enumerate(test):
	sample= sample.split("\t")
	sample = sample[0]
	print("\n======================================================================================================================\n")
	print("test_sample: ",sample)
	print("\n")
	conjunction_index=99 # number of sentences not exceeding 99 bcz of CC resolve
	sentences=sent_tokenize(sample)
	###Dividing input into Information and Question####
	information=[]
	questions=[]
	for i,sn in enumerate(sentences):
		if(re.search('\?' , sn)):
			questions.append(sn)
		else:
			information.append(sn)

	CC_resolve_pos_dict={}
	Original_pos_dict={}
	processed_information=[]
	ind_mod=0
	for i , inf in enumerate(information):
		cc_flag=0		
		if(re.search(r"[\$]" , inf)):
			new_curr_sent = Resolve_Currency(inf)
			inf = new_curr_sent
		#################POS Tagging of INFORMTION SENTENCE##############
		word_token = inf.split(" ")
		word_token = [X for X in word_token if X!=""]
		temp=nltk.pos_tag(word_token)
		resolve_pos_dict=dict(temp)
		CC_resolve_pos_dict[ind_mod]=resolve_pos_dict
		Original_pos_dict[ind_mod]=resolve_pos_dict
		#"""print("CC_resolve_pos_dict is ", CC_resolve_pos_dict)
		#print("\n\n")"""
		#Original_pos_dict = dict(map(reversed, CC_resolve_pos_dict.items()))
		#print("Original_resolve_pos_dict is ", Original_pos_dict)

		############################END OF POS TAGGING###############

		if (re.search(r"\band\b|\bif\b|\bbut\b|\bso\b", inf)):
			#print("Going to CC_Resolve module:",cc_flag)
			#print("CC resolve is " , CC_resolve_pos_dict)
			new_cc_sent = Resolve_Conjunctions(inf,ind_mod)
			conjunction_index=i
			ind_mod=ind_mod+1
			inf = new_cc_sent
			inf=inf.split(".")
			cc_flag=1
		if(cc_flag==1):
			for line in inf:
				processed_information.append(line)	

		else:
			processed_information.append(inf)

		ind_mod=ind_mod+1
			


	word_token = questions[0].split(" ")
	temp=nltk.pos_tag(word_token)

	CC_resolve_pos_dict[len(CC_resolve_pos_dict)]=dict(temp)
	processed_input=processed_information+questions
	Roles_Entities= Extract_Roles_entities(processed_input,CC_resolve_pos_dict,conjunction_index)
	#print("\n\n")
	#print("processed_information: ",processed_information)
	#print("\n\n")
	print ("Roles and Entities Dictionary" , Roles_Entities)
	Getting_values=Roles_Entities[len(Roles_Entities)-1]['Entity_Values']
	entity_name=Roles_Entities[len(Roles_Entities)-1]['Entities']
	#print("\n--------------------------------------------------\n")
	Getting_values = map(int, Getting_values)
	print("\n******** Answer Generation ***********\n")
	OUTPUT=Generate_Answer(test_sample_index, Getting_values)
	stanford_file_test=open("/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/stanford-parser-full-2018-10-17/TEST_DATA.txt","w")
	sample=sample.split("\t")
	stanford_file_test.write(str(sample[0]))
	stanford_file_test.close()

	frr=open("/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/stanford-parser-full-2018-10-17/TEST_DATA.txt","r")
	print("\n\n")
	################ Stanford parsing Retrieving ##############
	filen="/home/priyanka/Desktop/Final_NLP_Codes/Final_codes__/stanford-parser-full-2018-10-17/TEST_DATA.txt"
	res=subprocess.Popen(['bash','lexparser.sh',filen],stdout=subprocess.PIPE)
	#print("res:",res)
	out, err = res.communicate()
	#print("\n\n")
	#print(out)
	#print("\n-------------------------------------------\n")
	sent_parse_dict={}
	for i,test_input in enumerate(out.split("\n\n")):
		if(test_input!=''):
			elements=test_input.strip().split("\n")
			elements= [re.sub('[()]', "\t",s) for s in elements]
			temp={}
			for ele in elements:
				ele=ele.split("\t")
				key=ele[0]
				value=ele[1]
				value=value.split(",")
				value=[re.sub('-.*',"",e) for e in value]
				temp[key]=value

			sent_parse_dict[i]=temp

	question_parse=sent_parse_dict[len(sent_parse_dict)-1]
	#print("question_parse: ",question_parse)
	#print("\n")
	print("\n\n\nNATURAL LANGUAGE ANSWER GENERATION \n*********************\n")

	action=question_parse['root'][1]
	print("entity_name: ",entity_name)
	if('nsubj' in question_parse.keys()):
		subj=question_parse['nsubj'][1]
	if('amod' in question_parse.keys()):
		entity_name=question_parse['amod'][0]
				


	print(subj,action,OUTPUT,entity_name)
	print("\n======================================================================================================================\n")







