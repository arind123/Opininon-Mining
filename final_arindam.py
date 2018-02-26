#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:40:38 2017

@author: arindam
"""

import csv
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk import pos_tag
from nltk import pos_tag_sents
from nltk import word_tokenize
import re
from collections import Counter
from nltk.corpus import wordnet as wn
from sematch.semantic.similarity import WordNetSimilarity
from IPython.display import display
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

#######
xls_file = pd.ExcelFile('pm_qualities_data.xlsx')
print(xls_file.sheet_names)
df = xls_file.parse('Sheet1')
print(df['Comments'])
#Remove commas and full stops and make upper cases to lower
df['Comments'] = df['Comments'].str.lower()
df['Comments'] = df['Comments'].str.replace(',', '')
df['Comments'] = df['Comments'].str.replace('.', '')
print(df)
comments = df['Comments'].tolist()
print(comments[2])


tagged_comments = pos_tag_sents(map(word_tokenize, comments))
tagged_comments[31]
len(tagged_comments)

nou = [] 
useful_words = []
for i in range(0,38):
  tags = tagged_comments[i]
  nou = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'JJ' or pos == 'JJR' or pos == 'JJS' )]
  useful_words.append(nou) 

print(useful_words)
useful_words[2]
flat_list = [item for sublist in useful_words for item in sublist]
print(flat_list)
len(flat_list)
#Count frequency of words and sort in decreasing order
c = Counter(flat_list)
len(c)
print(c)
most_common_words = c.most_common()
most_common_words

#Find synonymous word for each of the unique words and create vocabulary 
uniques = sorted(set(flat_list))
print(uniques)
len(uniques)
synonyms = []
for i in range(0,len(uniques)):
 for syn in wn.synsets(uniques[i],wn.NOUN):
        synonyms.append(syn)
 
synonyms   
'''
# Find the hypernyms:
hypernyms = []
for i in range(0,len(uniques)):
 for syn in wn.synsets(uniques[i],wn.NOUN):
        hypernyms.append(syn.hypernyms())
        
hypernyms[0]       
y=wn.wup_similarity(synonyms[0],synonyms[2])
#Find the hyponyms
hyponyms = []
for i in range(0,len(uniques)):
 for syn in wn.synsets(uniques[i],wn.NOUN):
        hyponyms.append(syn.hyponyms())
        
hyponyms[0] 
'''
#Find the derivations and its synsets
derivations_names = []
for i in range(0,len(uniques)):
 for syn in wn.synsets(uniques[i],wn.NOUN):
    for l in syn.lemmas():
        for j in l.derivationally_related_forms():
            derivations_names.append(j.name())
 
len(derivations_names)    
derivations_names[1]  
unique_derivations = sorted(set(derivations_names))
print(unique_derivations)
len(unique_derivations)


derivations = []
for i in range(0,len(derivations_names)):
 for syn in wn.synsets(derivations_names[i],wn.NOUN):
        derivations.append(syn)
        
print(derivations)        



#Similarity matrix
all_words = synonyms + derivations
all_words = list(set(all_words))
for j in all_words:
    a = (j.name().split(".")[0])
    print(a)

print(len(all_words))

sim_matrix_synsets = [[wn.wup_similarity(w1, w2) for w1 in all_words] for w2 in all_words]
print(sim_matrix_synsets)
df_synsets = pd.DataFrame(sim_matrix_synsets, index=all_words,columns=all_words)
df_synsets.shape
display(df_synsets)
df_synsets.to_csv("test_wup_synsets.csv")





########

#####******Clustering*****#######

#######
def sumRow(matrix, i):
	return np.sum(matrix[i,:])

def buildSimilarityMatrix():
	matrix = np.zeros(shape=(len(all_words), len(all_words)))
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			dist = wn.wup_similarity(all_words[i], all_words[j])
			if dist > threshold:
				matrix[i,j] = 1
	return matrix

def determineRow(matrix):
	maxNumOfOnes = -1
	row = -1
	for i in range(len(matrix)):
		if maxNumOfOnes < sumRow(matrix, i):
			maxNumOfOnes = sumRow(matrix, i)
			row = i
	return row

def categorizeIntoClusters(matrix):
	groups = []
	while np.sum(matrix) > 0:
		group = []
		row = determineRow(matrix)
		indexes = addIntoGroup(matrix, row)
		groups.append(indexes)
		matrix = deleteChosenRowsAndCols(matrix, indexes)
	return groups

def addIntoGroup(matrix, ind):
	change = True
	indexes = []
	for col in range(len(matrix)):
		if matrix[ind, col] == 1:
			indexes.append(col)
	while change == True:
		change = False
		numIndexes = len(indexes)
		for i in indexes:
			for col in range(len(matrix)):
				if matrix[i, col] == 1:
					if col not in indexes:
						indexes.append(col)
		numIndexes2 = len(indexes)
		if numIndexes != numIndexes2:
			change = True
	return indexes

def deleteChosenRowsAndCols(matrix, indexes):
	for i in indexes:
		matrix[i,:] = 0
		matrix[:,i] = 0
	return matrix





threshold = 0.8#float(input("Threshold: "))

# build a matrix of similarity
mat = buildSimilarityMatrix()
mat.shape

groups = categorizeIntoClusters(mat)
print(groups)


with open('test.csv', 'w') as f:
     wtr = csv.writer(f, delimiter= ' ')
     wtr.writerows(groups)



groups = list(filter(any,groups))
groups
clusters =[]
for i in groups:
    clusters_temp = []
    for j in i:
        clusters_temp.append(all_words[j])
    clusters.append(clusters_temp)    

with open('Clustered_words.csv', 'w') as f:
     wtr = csv.writer(f, delimiter= ' ')
     wtr.writerows(clusters)
     
     
     
len(clusters)     
len(clusters[1])
clusters = list(filter(any,clusters))


names =[]    
for i in clusters:
    s = []
    u = []
    for j in i:
        s.append(j.name().split(".")[0])
        u = sorted(set(s))
    names.append(u)
    
with open('Clustered_final.csv', 'w') as f:
     wtr = csv.writer(f, delimiter= ' ')
     wtr.writerows(names)    
     
     
