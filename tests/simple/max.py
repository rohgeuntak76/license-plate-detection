
candidates = [('abcd',0.1),('zdf',None)]

valid = [(text,score) for text,score in candidates if score is not None]

lic_text, lic_score = max(valid,default=(None,None))

print(lic_text)
print(lic_score)