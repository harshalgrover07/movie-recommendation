# %%
#pip install gensim -q

# %%
# Lets import thr libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
from gensim.models import  Word2Vec
from gensim import downloader as gen_download
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

# %%
# Lets load the data set
data = pd.read_csv(r'movies.csv')

# %%
data.head()

# %%
data.shape

# %%
# Check for missing values

data.isnull().sum()

# %%
# Lets convert text to tokens
data['genre_tokens']= data['genres'].str.lower().str.split(pat='|')

# %%
data

# %%
embed_model = Word2Vec(sentences=data['genre_tokens'],window=2,sg=1,vector_size=20)

# %%
vocabulary  = embed_model.wv.index_to_key

# %%
vocabulary

# %%
# We need to remove the movies where genre is not provided

data[data['genres']=='(no genres listed)'].shape

# %%
drop_index = data[data['genres']=='(no genres listed)'].index
data.drop(drop_index,inplace=True)
data.reset_index(inplace=True)

# %%
# Lets train SkipGram model on our data and genrate word vectors for tokens

embed_model = Word2Vec(sentences=data['genre_tokens'],window=2,sg=1,vector_size=20)

# %%
embed_model.wv['children']

# %%

embed_model.wv.most_similar('children',topn=5)

# %%
embed_model.wv.most_similar('romance',topn=5)

# %%
embed_model.wv.most_similar('horror',topn=5)

# %%
embed_model.wv.key_to_index

# %%
# Lets try some pretrained model (Glove)

model_glove = gen_download.load('glove-twitter-25')

# %%
model_glove['horror']

# %%
model_glove['documentary']

# %%
# Lets convert these tokens to word embedding

data['genre_embeddings']=data['genre_tokens'].apply(lambda x: [embed_model.wv[w] for w in x])

# %%
data

# %%
data['genre_embeddings'][0]

# %%
data['genre_avg_embeddings'] = data['genre_embeddings'].apply(lambda x: np.mean(x,axis=0))

# %%
data

# %%
data['genre_avg_embeddings'][0]

# %% [markdown]
# ## Recomendation System

# %%
# create a similarity matrix

simi_matrix = cosine_similarity(data['genre_avg_embeddings'].to_list())

# %%
simi_matrix

# %%
def recommender(selected_movie,nos=5):
  if selected_movie in data['title'].values:
    idx = data[data['title']==selected_movie].index[0]
    top_n_idx = simi_matrix[idx].argsort()[::-1][1:nos+1]
    for i in top_n_idx:
      print(data['title'].iloc[i])
  else:
      print('Movie not Found')


# %%
movie_name = 'Black Butler: Book of the Atlantic (2017)'
data[data['title']==movie_name].index[0]

# %%

simi_matrix[9703].argsort()[::-1][1:6]

# %%
recommender('Jumanji (1995)')

# %%


# %%



