
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

f = open('asci.txt')
codelines = f.readlines()


# In[3]:

codelines


# In[4]:

jobsDict = {}
province = ''
subprovince = ''
subprovince_flag = False
for line in codelines:
    if line != '\n':
        job_description = line.split("=")
        print(job_description)
        if job_description[0] == 'Provinces':
            province = job_description[1].replace("\n", "")
            subprovince_flag = False
        elif job_description[0] == 'Sub-Provincial':
            subprovince = job_description[1].replace("\n", "")
            subprovince_flag = True
        else:
            if len(job_description) > 1:
                if subprovince_flag:
                    job_value = [province, subprovince, job_description[1].replace("\n", "")]
                else:
                    job_value = [province, '', job_description[1].replace("\n", "")]
                jobsDict[job_description[0]] = job_value
            else:
                jobsDict[job_description[0]] = []
        


# In[6]:

jobsDict


# In[21]:

provinces = [[]]
temp = []

#for line in codelines:
#    print line
    
for line in codelines:
    if (line[0] == '+'):
        provinces.append(temp)
        temp = []
        temp.append(line)
        
    else:
        temp.append(line)



# In[23]:

len(provinces)


# In[ ]:



