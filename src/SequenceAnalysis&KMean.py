
# coding: utf-8

# In[1]:

import pdb
import numpy as np
import pandas as pd
import pickle
from __future__ import print_function
from tqdm import *


# In[ ]:

# Retrieve cost spreadsheets
#costs_sub = pd.DataFrame.from_csv('./SubstitutionCosts_v2.csv')
#costs_del = pd.DataFrame.from_csv('./SubstitutionCosts.csv')
#costs_ins = pd.DataFrame.from_csv('./SubstitutionCosts.csv')

costs = pd.DataFrame.from_csv('./Costs_v4.csv')


# In[ ]:

# Cost functions which returns cost from lookup in costmartix 
# We will update the definition once we have the cost matrix

def costDeletion(s):
    return costs.get_value(s,'Del')

def costInsertion(s):
    return costs.get_value('Ins',s)

def costSubstitution(s1, s2):
    return costs.get_value(s1,s2)


# In[ ]:

# Does the sequence analysis between job-sequences of two different persons
def OptimalMatching(s1, s2):
    
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    cost = np.zeros((lenstr1 + 1, lenstr2 + 1), dtype=int)
    
    # initialization : comparaison with a null sequence
    min_cost = 1
    for i in range(lenstr1 + 1):
        cost[i, 0] = i*min_cost
    for j in range(lenstr2 + 1):
        cost[0, j] = j*min_cost

    for el1 in range(lenstr1):
        for el2 in range(lenstr2):
            if s1[el1] == s2[el2]:
                cost[el1 + 1, el2 + 1] = cost[el1, el2] # cost = 0 because they are the same
            else:
                cost[el1 + 1, el2 + 1] = min(
                                           cost[el1, el2 + 1] + costDeletion(s2[el2]), # deletion
                                           cost[el1 + 1, el2] + costInsertion(s1[el1]), # insertion
                                           cost[el1, el2] + costSubstitution(s1[el1],s2[el2]) # substitution
                                          )
    return cost[lenstr1, lenstr2]


# In[ ]:

#Input is career matrix of all the persons
# Does the sequence analysis between everypair and returns a Distance Matrix.
def SequenceAnalysis(careers):
    maxLen = len(careers)
    costTable = np.zeros((maxLen, maxLen), dtype=int)
    #for i in range(len(careers))):
    for i in tqdm(range(len(careers))):
        for j in range(len(careers)):
            costTable[i][j] = OptimalMatching(careers[i],careers[j])
    return costTable


# In[ ]:

new_trajs = pickle.load(open('./recoded_trajs.p','rb'))


# In[ ]:

# These are for
table = SequenceAnalysis(new_trajs)


# In[18]:

pickle.dump(table,open('pairwise_v4.p','wb'))
#table = pickle.load(open('pairwise_v2.p','rb'),encoding='latin1')
table


# **Functions used for clustering (below)**. Takes three arguments- DistanceMatrix after Sequence Analysis, number of clusters, number of iterations

# In[19]:

def compute_average_distance(i,j,list_clusters):
    list_cluster_elements = list_clusters[j]
    distance = []
    for p in list_cluster_elements:
        distance.append(table[i][p])
    if float(len(distance))!= 0:
        return sum(distance) / float(len(distance))
    else:
        print("Distance Zero")
        return sum(distance)

def distribute_initialClusters(no_left_rows,no_clusters,list_clusters):
    for i in range(no_left_rows):
        cluster_similarity = np.zeros(no_clusters,np.int)
        for j in range(no_clusters):
            cluster_similarity[j] = compute_average_distance(i,j,list_clusters)
        cluster_belong = np.argmin(cluster_similarity)
        list_clusters[cluster_belong].append(no_clusters+i)

def clustering(table, no_clusters, no_iterations):
    list_clusters = []#{}
    for i in range(no_clusters):
        tempList = []
        tempList.append(i)
        list_clusters.append(tempList)#list_clusters[i] = tempList
    no_rows = table.shape[0]
    no_left_rows = no_rows - no_clusters
    distribute_initialClusters(no_left_rows,no_clusters,list_clusters)
    for p in tqdm(range(no_iterations)):
        for i in range(no_rows):
            cluster_similarity = np.zeros(no_clusters,np.int)
            earlier_cluster = -1
            for j in range(no_clusters):
                temp = list_clusters[j]
                if i in temp:
                    earlier_cluster = j
            for j in range(no_clusters):
                cluster_similarity[j] = compute_average_distance(i,j,list_clusters)
            cluster_belong = np.argmin(cluster_similarity)
            list_clusters[cluster_belong].append(i)
            list_clusters[earlier_cluster].remove(i)
        if sum([len(list_clusters[i]) for i in range(no_clusters)]) != no_rows:
            print("Error: More number of elements in cluster")
            print("Iteration p")
            break
    return list_clusters


# List Clusters below contain the final clusters as a dictionary

# In[39]:

#Results are for v5:
list_clusters4 = clustering(table,4,500)


# In[37]:

#list_clusters3 = list_clusters

for cluster in list_clusters4:
    print(len(cluster))


# In[38]:

pickle.dump(list_clusters4,open('clusters_v4_4.p','wb'))

