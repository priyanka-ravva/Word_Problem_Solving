import nltk
from nltk import word_tokenize,sent_tokenize
import re
import itertools
from nltk.stem.wordnet import WordNetLemmatizer

######Modules for Pre-Processing###########

#####Module1 Resolve_Currency#############
def Resolve_Currency(inf):
	print ("Sent received at Currency module is:\n",inf)
	word_token = inf.split(" ")
	mod_sentence=""
	print("\n")
	for token_curr in word_token:
		if(re.search(r"\$",token_curr)):
			mod_sentence=mod_sentence+" "+token_curr[1:]+" "+"dollars"
		elif(re.search(r"\₹",token_curr)):
			mod_sentence=mod_sentence+" "+token_curr[1:]+" "+"rupees"
		else:
			mod_sentence=mod_sentence+" "+token_curr
	print("Modified sentence at Currency module is:\n",mod_sentence[1:])
	print("\n")
	return(mod_sentence[1:])

def Resolve_Conjunctions(inf,i):
	
	
	###Applying POS tagging before splitting the sentence#####
	word_token = inf.split(" ")
	print("Word Tokens ", word_token)
	pos_tags_sent=[]
	temp=nltk.pos_tag(word_token)
	pos_tags_sent.append(temp)
	print("\n\n")
	print(pos_tags_sent)
	print("\n\n")
	resolve_pos_dict=dict(pos_tags_sent[0])
	CC_resolve_pos_dict[i]=resolve_pos_dict
	print("CC_resolve_pos_dict is ", CC_resolve_pos_dict)
	print("\n\n")
	new_sent=re.split(r'(?:and|if|but)', inf)
	print("CC Resolve sent is ", new_sent)
	print("\n")
	mod_sentence=''
	for sent in new_sent:
		mod_sentence+=sent+'.'
		
	print("Modified sentence at Conjunction Resolve module is:\n",mod_sentence[:-2])
	return mod_sentence[:-2]

def Extract_Roles_entities(processed_information,CC_resolve_pos_dict,conjunction_index):

	dict_cc_resolve={}

	####Word Tokenizing the information sentences####
	tokens=[word_tokenize(new_sent_sample) for new_sent_sample in processed_information]
	print("\n----------------tokens -------------------\n")
	print("tokens:",tokens)
	print("\n\n-------------- POS tags ----------------\n")


	#iterate over the all sentences:
	for i,sent in enumerate(tokens):
		print("\n")
		print(sent,conjunction_index,i)
		print("\n--------------------------\n")
		temp={}

		mod_I=i
		if(i==conjunction_index+i):
			#mod_I=i
			if (i!=0):
				i=i-1

		for j,w in enumerate(sent):
			
			print(j,len(sent),w,i,mod_I)
			print("CC_resolve_pos_dict[i]: ",CC_resolve_pos_dict[i])
			if(CC_resolve_pos_dict[i][w] in verb_tags):
				Owners1=[]
				Owners2=[]
				print("IN IF")
				verb=[w]
				if(j!=0):
					pre_verb=sent[:j]
					pre_verb_tags = [CC_resolve_pos_dict[i][w] for w in sent[:j]]
					#print("PREverb is" , pre_verb)
					for o,tag in enumerate(pre_verb_tags):
						print("PREverb is" , pre_verb, tag)
						if (tag in Noun_tags):
							print("In Noun tags ",sent[:j],sent[o])
							Owners1.append(sent[o])
							print("Owners1 is ", Owners1)
						if (tag in pronoun_tags):
							print("In Pronoun Tag")
							Curr_Owners=dict_cc_resolve[mod_I-1]["Owners1"]
							print("***************",mod_I, Curr_Owners[0]) 
							pre_verb=dict_cc_resolve[mod_I-1]["pre_verb"]
							Owners1.append(Curr_Owners[0])
							print("Owners1 2nd Prp is ", Owners1)
					
					#if (pre_verb in 
				else:
					print("In else")
					if(mod_I!=0):
						pre_verb = dict_cc_resolve[mod_I-1]["pre_verb"]
					else:
						pre_verb=None
				print(sent[j+1:])
				after_verb_tags=[CC_resolve_pos_dict[i][w] for w in sent[j+1:]]
				print("after_verb_tags: ",after_verb_tags)
				Averb=[]
				prp_verb=[]
				Entities=[]
				Prp_verb_Tags=[]
				#####Storing the entity vlues:
				for k,tag in enumerate(after_verb_tags):
					#print("Tag is ", tag)
					if (tag in object_tag):
						Entities.append(sent[j+1+k])
					if (tag in Noun_tags):
						print("In Owner2 tags", sent[j+k+1])
						Owners2.append(sent[j+k+1])
					if not (tag in Prep_tags):
						Averb.append(sent[j+1+k])
					else:
						prp_verb=sent[j+1+k:]
						Prp_verb_Tags=[CC_resolve_pos_dict[i][w] for w in sent[j+1+k:]]
						break
				for o2,tag in enumerate(Prp_verb_Tags):
					if (tag in object_tag):
						Entities.append(prp_verb[o2])
					if (tag in Noun_tags):
						print("In Owner2 tags", sent[j+o2+1])
						Owners2.append(prp_verb[o2])
				if (Entities==[] and mod_I!=0):
					Entities = dict_cc_resolve[mod_I-1]["Entities"]
				
					

				after_verb=Averb
			
				print("Averb: ",Averb)
				print("Prp_Verb: ",prp_verb)
				print("Entities: ",Entities)
				print("PREverb is" , pre_verb)
				print("Owners1 is", Owners1)
				print("Owners2 is", Owners2)
			else:
				if(j==len(sent)-1 and i!=0 and verb==[]):
					print("IN elseF")
					print("Verb is not in the sentence: \n")
					print("prev_sent,Current_sent : ", i , tokens[i-1],sent)
					print("\n")
					print("Dict is ", dict_cc_resolve[i-1], j)
					pre_verb = dict_cc_resolve[i-1]["pre_verb"]
					Owners= dict_cc_resolve[i-1]["Owners"]
					verb= dict_cc_resolve[i-1]["verb"]
					after_verb_tags=[CC_resolve_pos_dict[i][w] for w in sent]
					print("after_verb_tags: ",after_verb_tags)
					Averb=[]
					prp_verb=[]
					Prp_verb_Tags=[]
					for k,tag in enumerate(after_verb_tags):
						if (tag in object_tag):
							Entities.append(sent[j+1+k])
						if not (tag in Prep_tags):
							Averb.append(sent[k])
						else:
							prp_verb=sent[k:]
							Prp_verb_Tags=[CC_resolve_pos_dict[i][w] for w in sent[k:]]
							break
					for o2,tag in enumerate(Prp_verb_Tags):
						if (tag in object_tag):
							Entities.append(prp_verb[o2])
						if (tag in Noun_tags):
							print("In Owner2 tags", sent[j+o2+1])
							Owners2.append(prp_verb[o2])
					
					if (Entities==[]):
						Entities = dict_cc_resolve[i-1]["Entities"]				

					after_verb=Averb

		print("\n\n")
		temp["verb"]=verb
		temp["pre_verb"]=pre_verb
		temp["after_verb"]=after_verb	
		temp["prp_verb"]=prp_verb
		temp["Entities"]=Entities
		temp["Owners1"]=Owners1
		temp["Owners2"]=Owners2

		dict_cc_resolve[mod_I]=temp
	
		i=mod_I

	print("\n")
	print("dict_cc_resolve= ",dict_cc_resolve)




####Pre-Processing the Input#################

#CC_input2="John has 50 apples and ate 3 of them . How many apples does he have ?"
CC_input2="Joan found 25 seashells on the beach and she gave some of her seashells to Sam . She has 10 seashells . How many seashells did she give Sam ?"



Original_pos_dict={}
CC_resolve_pos_dict={}

sentences=sent_tokenize(CC_input2)
print("sentences: ",sentences)

###Dividing input into Information and Question####
information=[]
questions=[]
for i,sn in enumerate(sentences):
	if(re.search('\?' , sn)):
		questions.append(sn)
	else:
		information.append(sn)
			
	
print("sentences information : ")
print(information)
print("\n")
print("Question Related Sentences : ")
print(questions)
print("\n")

conjunction_keywords=["and","but","if","then"]
currency_keywords=["₹","$"]


verb_tags=['VBD','VBZ','VBP','VBN','VBG','VB']
Noun_tags=['NNP','NN']
Prep_tags=['IN','TO']
object_tag=['NNS','NNPS']
pronoun_tags=['PRP','PRP$']



######### Schemas ##############
schemas_keys={}
schemas_keys['Change Out']=["put", "place", "plant", "add", "sell", "distribute", "load", "give"]
schemas_keys['Combine']=["together", "in all", "combined", "in total", "total"]
schemas_keys['Reduction']=["eat", "destroy", "spend", "remove", "decrease"]
schemas_keys['Change In']=["take from", "get", "pick", "buy", "borrow", "steal"]


conjunction_index=99 # number of sentences not exceeding 99 bcz of CC resolve



processed_information=[]
for i , inf in enumerate(information):
	cc_flag=0
	
	if(re.search(r"[\$\₹]" , inf)):
		print("Going to currency module:")
		new_curr_sent = Resolve_Currency(inf)
		inf = new_curr_sent
	#################POS Tagging of INFORMTION SENTENCE##############
	word_token = inf.split(" ")
	#pos_tags_sent=[]
	temp=nltk.pos_tag(word_token)
	#pos_tags_sent.append(temp)
	print("\n\n")
	#print(pos_tags_sent)
	print("\n\n")
	resolve_pos_dict=dict(temp)
	CC_resolve_pos_dict[i]=resolve_pos_dict
	Original_pos_dict[i]=resolve_pos_dict
	print("CC_resolve_pos_dict is ", CC_resolve_pos_dict)
	print("\n\n")
	#Original_pos_dict = dict(map(reversed, CC_resolve_pos_dict.items()))
	print("Original_resolve_pos_dict is ", Original_pos_dict)

	############################END OF POS TAGGING###############

	if(re.search(r'(?:and|if|but)', inf)):
		print("Going to CC_Resolve module:",cc_flag)
		new_cc_sent = Resolve_Conjunctions(inf,i)
		conjunction_index=i
		inf = new_cc_sent
		inf=inf.split(".")
		cc_flag=1
	print("cc_flag: ",cc_flag)
	print("-------------------")
	if(cc_flag==1):
		for line in inf:
			processed_information.append(line)	

	else:
		print("*** cc else case conjunction index...",conjunction_index,cc_flag)
		processed_information.append(inf)
		
Roles_Entities= Extract_Roles_entities(processed_information,CC_resolve_pos_dict,conjunction_index)


print("\n\n")
print("processed_information: ",processed_information)
print("\n\n")


