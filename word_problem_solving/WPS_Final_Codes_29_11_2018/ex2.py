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
