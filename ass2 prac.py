import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Data
m = pd.DataFrame({
 'title':['Shawshank','Godfather','Dark Knight','Pulp Fiction'],
 'genres':['Drama','Crime, Drama','Action, Drama','Crime'],
 'desc':['Two men bond in prison.','Crime family legacy.','Batman vs Joker.','Mob hitmen stories.']
})

m['content'] = m['genres'] + ' ' + m['desc']

# TF-IDF + Similarity
tf = TfidfVectorizer(stop_words='english')
mat = tf.fit_transform(m['content'])
sim = linear_kernel(mat, mat)

# Recommend Function
def rec(t):
    i = m[m.title==t].index[0]
    s = sorted(enumerate(sim[i]), key=lambda x:x[1], reverse=True)[1:3]
    return m.title[[j for j,_ in s]]

print(rec('Pulp Fiction'))
