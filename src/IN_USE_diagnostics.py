
# coding: utf-8

# In[1]:

import pandas as pd
from collections import Counter


# In[2]:

df = pd.read_csv("cc_data_jobrecoded07_12.csv",dtype=str)
# We need to NOT load numbers as floats, 2000 is not the same code as 2000.0
# pandas reads in blank cells as NaN. This is useful.


# ** Diagnostics with just the job sequences:**

# In[3]:

# Career sequences (with dates)
jobs = df.ix[:,48:]


# In[4]:

# Get rid of the date columns
def remove_date_cols(dataframe):
    date_inds = []
    for i in range(len(dataframe.columns)):
        label = dataframe.columns[i]
        if ('s' in label or 'e' in label) and ('job' in label):
            date_inds.append(i)
    dataframe = dataframe.drop(jobs.columns[date_inds],axis=1)
    return dataframe


# In[5]:

jobs = remove_date_cols(jobs)


# In[7]:

# Get all the job codes in the data
import math
data_jobs = jobs.values.flatten()
#print len(data_jobs)\
#Remove the NaNs
data_jobs = [x for x in data_jobs if x==x]
#print len(data_jobs)
#Convert the floats to strings, because 2001 is not the same job as 2001.1


# In[8]:

#Opening codebook
import re
f = open('codebook_regions.txt')
codelines = f.readlines()
region_jobs = []
for line in codelines:
    region_jobs.append(re.findall(r'\d+\.\d+', line))
# Purge empty lists from lines with no numbers
region_jobs = [item for sublist in region_jobs for item in sublist]


# In[50]:

data_jobs = set(data_jobs)
region_jobs = set(region_jobs)


# In[51]:

print len(region_jobs)
print len(data_jobs)


# In[52]:

both = (data_jobs.intersection(region_jobs))
len(both)


# In[53]:

mil_jobs =[job for job in data_jobs if job[:2] == '35']
len(mil_jobs)


# How many people have region-specific jobs in their histories?

# In[98]:

jobs in region_jobs


# ** Let's do some tests with gender: **

# In[59]:

Counter(df['gender'])


# In[78]:

women = df.loc[df['gender'] == '0']
men = df.loc[df['gender'] == '1']
women_jobs = women.ix[:,48:]
men_jobs = men.ix[:,48:]
women_jobs.count(axis=1)


# In[ ]:



