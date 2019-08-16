import nltk
from nltk import word_tokenize,sent_tokenize
import re
import itertools

#CC_input="John has 5 apples and ate 2 of them . Sam had 49 pennies and 34 nickels in his bank."
#CC_input2="John has 5 apples and ate 2 of them"
#CC_input2="John has $5 and Mary gave $2 to him"
#CC_input2="John had 5 apples and Mary gave John 3 apples"
CC_input2="John has 5 apples . He gave 3 to Mary"

sentences=sent_tokenize(CC_input2)
print("sentences: ",sentences)

conjunction_keywords=["and","but","if","then"]
currency_keywords=["₹","$"]





word_token=[sen.split(" ") for sen in sentences]
print("word_token: ",word_token)
print("\n")
#########Resolving Currency######
Dummy_new_sent=[]
mod_sentence=""

for i in range(len(word_token)):
	for token_curr in word_token[i]:
		print(token_curr)
		if(re.search(r"\$",token_curr)):
			print("here")
			Dummy_new_sent.append(token_curr[1:])
			Dummy_new_sent.append("dollars")
			mod_sentence=mod_sentence+" "+token_curr[1:]+" "+"dollars"
		elif(re.search(r"\₹",token_curr)):
			Dummy_new_sent.append(token_curr[1:])
			Dummy_new_sent.append("rupees")
			mod_sentence=mod_sentence+" "+token_curr[1:]+" "+"rupees"
		else:
			Dummy_new_sent.append(token_curr)
			mod_sentence=mod_sentence+" "+token_curr
	
		

	
print("mod_sentence: ",mod_sentence)
modified_sent=sent_tokenize(mod_sentence)
print("input sentences:",modified_sent)
print("\n")




############POS Tagging ############
pos_tags_sent=[]
temp=nltk.pos_tag(Dummy_new_sent)
pos_tags_sent.append(temp)

print(pos_tags_sent)
print("\n\n")


exit()

new_sent=[re.split('and',sent) for sent in modified_sent]
#new_sent=[re.split("[/\b['but','and']\b/]",sent) for sent in sentences]
#converting into one list:
new_sent=list(itertools.chain(*new_sent))
print("List of Sentences: ",new_sent)




#new_sent=Dummy_new_sent
tokens=[word_tokenize(new_sent_sample) for new_sent_sample in new_sent]
print("\n----------------tokens -------------------\n")
print("tokens:",tokens)
print("\n\n-------------- POS tags ----------------\n")



verb_tags=['VBD','VBZ','VBP','VBN','VBG','VB']
Noun_tags=['NNP','NNS','NN','NNPS']
Prep_tags=['IN','TO']
object_tag=['NNS','NNPS']

dict_pos_sent=dict(pos_tags_sent[0])

print(dict_pos_sent)

dict_cc_resolve={}

#iterate over the all sentences:
for i,sent in enumerate(tokens):
	print("\n")
	print(sent)
	print("\n--------------------------\n")
	temp={}
	for j,w in enumerate(sent):
		print(j,len(sent))
		if(dict_pos_sent[w] in verb_tags):
			print("IN IF")
			verb=[w]
			if(j!=0):
				pre_verb=sent[:j]
			else:
				if(i!=0):
					pre_verb = dict_cc_resolve[i-1]["pre_verb"]
				else:
					pre_verb=None
			print(sent[j+1:])
			after_verb_tags=[dict_pos_sent[w] for w in sent[j+1:]]
			print("after_verb_tags: ",after_verb_tags)
			Averb=[]
			prp_verb=[]
			for k,tag in enumerate(after_verb_tags):
				if not (tag in Prep_tags):
					Averb.append(sent[j+1+k])
				else:
					prp_verb=sent[j+1+k:]
					break			

			after_verb=Averb
			
			print("Averb: ",Averb)
			print("Prp_Verb: ",prp_verb)
		
		else:
			if(j==len(sent)-1 and i!=0 and verb==[]):
				print("IN elseF")
				print("Verb is not in the sentence: \n")
				print("prev_sent,Current_sent : ", i , tokens[i-1],sent)
				print("\n")
				print("Dict is ", dict_cc_resolve[i-1], j)
				pre_verb = dict_cc_resolve[i-1]["pre_verb"]
				verb= dict_cc_resolve[i-1]["verb"]
				after_verb_tags=[dict_pos_sent[w] for w in sent]
				print("after_verb_tags: ",after_verb_tags)
				Averb=[]
				prp_verb=[]
				for k,tag in enumerate(after_verb_tags):
					if not (tag in Prep_tags):
						Averb.append(sent[k])
					else:
						prp_verb=sent[k:]
						break			

				after_verb=Averb
			print("\n")
			"""pos_tag_line=nltk.pos_tag(sent)
			dict_pos_tag_line=dict(pos_tag_line)
			print(dict_pos_tag_line)
			print("\n\n")
			for l,(key,tag) in enumerate(dict_pos_tag_line.items()):
				print(key,tag)
				if(tag=='CD'):
					print("CD value: ",l,tag,key)
				if(tag in Noun_tags):
					print("subj/obj Noun : ",l,tag,key)
				print("\n")"""



	print("\n\n")
	temp["verb"]=verb
	temp["pre_verb"]=pre_verb
	temp["after_verb"]=after_verb	
	temp["prp_verb"]=prp_verb
	dict_cc_resolve[i]=temp


print("\n")
print("dict_cc_resolve= ",dict_cc_resolve)

