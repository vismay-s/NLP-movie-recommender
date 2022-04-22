# -*- coding: utf-8 -*-
"""
Created on Sat April 18 23:08:53 2020

@author: vismay sudra
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv(r'C:\zminiproject\data Analysis\movie_dataset.txt')
features = ['keywords','cast','genres','director']


def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
for feature in features:
    df[feature] = df[feature].fillna('')
df["combined_features"] = df.apply(combine_features,axis=1)
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

index_list=[] 
pers_score=[]  
tupList=[]     
perslist=[]   

movie_like = str(input('Enter movie'))
movie_liked = movie_like.strip()
movie_index = get_index_from_title(movie_liked)
sim_val=cosine_sim[movie_index]
similar_movies =  list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
i=0
for element in sorted_similar_movies:
    a=element[0]
    index_list.append(a)
    Pscore= sim_val[a] + (df.vote_average[a]/15)
    pers_score.append(Pscore)
    i=i+1
    if i>=20:
        break
for j in range (20):
    tupList=[(index_list[j]),pers_score[j]]
    perslist.append(tupList)
    
Personalized_list = sorted(perslist, key=lambda perslist:perslist[1],reverse=True)
k=0

print("Top 10 movies similar to "+movie_liked+" are:\n")
for element in Personalized_list:
    print(get_title_from_index(element[0]))
    k=k+1
    if k>=10:
        break
