import nltk
from nltk import word_tokenize,sent_tokenize
import re
from nltk.stem.wordnet import WordNetLemmatizer
words = ['gave','went','going','dating',"said","talking","ate","eaten","skipping","go","wants"]

for word in words:
    print word+"-->"+WordNetLemmatizer().lemmatize(word,'v')
print("\n\n")



schemas_keys={}
schemas_keys['Change Out']=["put", "place", "plant", "add", "sell", "distribute", "load", "give"]
schemas_keys['Combine']=["together", "in all", "combined", "in total", "total"]
schemas_keys['Reduction']=["eat", "destroy", "spend", "remove", "decrease"]
schemas_keys['Change In']=["take from", "get", "pick", "buy", "borrow", "steal"]

user_input="The boys have 3 red apples. mary has 4 green apples. get the number of red apples they have together? "

sent=sent_tokenize(user_input)
print("Input: ")
print(sent)
print("\n")


information=[]
questions=[]
for i,sn in enumerate(sent):
	print(sn)
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

#Identifying the Schema
question_tokens=word_tokenize(questions[0])


print("schema Identification.....")
possible_schemas=[]
for schema,values in schemas_keys.iteritems():
	for w in question_tokens:
		if(w in values):
			possible_schemas.append(schema)
		
print(possible_schemas)
print("\n")

tokens=[]
tokens.append([word_tokenize(information[i]) for i in range(len(information))])
print("tokens:",tokens)
print("\n")

sent_own_obj_dict={}
for i,line_tokens in enumerate(tokens[0]):
	pos_tags_line=nltk.pos_tag(line_tokens)
	print("pos_tags_line: ",pos_tags_line)
	owner_tags=['NNP','NNS','NN','NNPS']
	verb_tags=['VBD','VBZ','VBP','VBN','VBG','VB']
	object_tags=['NNS','NNPS','NN']

	owner=[]
	verbs=[]
	objects=[]
	adj_word=''
	temp_val=''

	for j,w in enumerate(pos_tags_line):
		if(w[1] in owner_tags):
			owner.append(w)

		if(w[1] in object_tags):
			if((pos_tags_line[j-1][1])=='JJ'):
				adj_word=pos_tags_line[j-1][0]
				word=adj_word+"-"+w[0]
				print("word:" ,word,w) # ALert
				w_mod=(word,w[1])
				objects.append(w_mod)

			else:
				objects.append(w)
				
		if(w[1] in verb_tags):
			verbs.append(w)
		if(w[1]=='CD'):
			entity_value=w
			temp_val=w[0]


	if not (temp_val):
		# dummy sentences....no use of that sentence
		del information[i]
		continue




	k1="owner"+str(i)
	k2="object"+str(i)
	k3="entity_value"+str(i)
	k4="verb"+str(i)

	v1=owner[0][0]
	v2=objects[1][0]
	v4=verbs[0][0]


	temp={}
	temp[k1]=v1
	temp[k2]=v2
	temp[k3]=entity_value[0]
	temp[k4]=v4
	sent_own_obj_dict[i]=temp
	 
		




print("\n\n")
print(sent_own_obj_dict)
print("\n\n")


#exit()

question_tokens_tags=nltk.pos_tag(question_tokens)
print(question_tokens_tags)

pronoun_tag=['PRP','NN','NNP']
verb_tag=['VBP','VBZ','VBD']
object_tag=['NNS','NNPS']




adj_word=''
# If more than number of possiblities occurs then it will take last one....IN case of Verbs
for i,w in enumerate(question_tokens_tags):
	if(w[1] in pronoun_tag):
		start=w[0]
	if(w[1] in verb_tag):
		second=w[0]

	if((question_tokens_tags[i-1][1])=='JJ'):
		adj_word=question_tokens_tags[i-1][0]
		last=adj_word+"-"+w[0]
		print("If last",last,w)



if not(start):
	start="They"
if not(last):
	last=sent_own_obj_dict[0]["object0"]

#print("=============> ",start,second,last)


def Combine(sent_own_obj_dict):
	values=[]
	for i in range(len(information)):
		x="entity_value"+str(i)
		#print("x",x)
		xx="object"+str(i)

		if(last==sent_own_obj_dict[i][xx]):
			values.append(int(sent_own_obj_dict[i][x]))	
	#print(values)
	result=sum((values))
	print("answer: ",result)
	third=result
	print("Answer in Natural Language Processing....")
	print(start+" "+second+" "+str(third)+" "+last)

Combine(sent_own_obj_dict)
