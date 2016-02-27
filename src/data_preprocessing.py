
# coding: utf-8

# In[1]:

import pandas as pd


# In[3]:

f = open('codebook.txt')
codelines = f.readlines()


# In[4]:

provinces = codelines.split('+')


# In[13]:

provinces = [[]]
temp = []

#for line in codelines:
#    print line
    
for line in codelines:
    if (line[0] == ):
        provinces.append(temp)
        print "I'm here"
        
    else:
        temp.append(line)



# In[ ]:



