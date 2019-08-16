import re

inf="David gave 3 candies to John and Ruth butter gave 2 apples to David ."
#int="the book I gave"
#if(re.search(r"\b[and|if|but]\b", inf)):
if (re.search(r"\band\b|\bif\b|\bbut\b", inf)):
	#if(re.search('/\b[tT]he\b/', inf)):
	print("CC present")
	new_sent=re.split(r"(?:\band\b|\bif\b|\bbut\b)", inf)
	print("CC Resolve sent is ", new_sent)
	
