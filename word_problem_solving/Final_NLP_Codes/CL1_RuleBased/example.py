import sys, os
os.system
import subprocess
from subprocess import Popen, PIPE
import re



filen="/home/priyanka/Desktop/Word_Problem_solving/stanford-parser-full-2018-10-17/input_new2.txt"
#line="teenu is good girl"
res=subprocess.Popen(['bash','lexparser.sh',filen],stdout=subprocess.PIPE)
#print("res:",res)
out, err = res.communicate()
print("\n\n")
print(out)
print("\n-----------\n")

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
	




print("sent_parse_dict: ",sent_parse_dict)

#subprocess.call(['bash','lexparser.sh',filen])
