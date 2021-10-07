#!/usr/bin/env python
# coding: utf-8

# 

# In[29]:


import numpy as np
import string
import re
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
#from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from timeit import default_timer as timer


# # Sentence And Functions

# In[30]:


sentence = ['The dogs are fighting over a bone.', 'Annie is very smart.', 'dogs do fight at times',
            'My teamates are very excited about our project', 'Alice loves wonderland', 'Wonderland is beautiful',
            'Good work on your assignment.', 'AI is taking over the world', 'ML is also taking over the world',
            "Artificial intelligence is taking over the world","Joe is very smart",
            "John did a very bad job.", "Mary is horrible at her job", "Excellent work",
            "Good job with your homework",  "pencils are good for writing",
            "Saint thomas is an excellent school", "Saint thomas has a beautiful campus","Kweweli is a lucky guy", 
            'Mary hope to visit wonderland one day']

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN)


# # Process Text 

# In[31]:


# PROCESS TEXT
# prepare regex for char filtering
re_punc = re.compile('[%s]'% re.escape(string.punctuation))

# remove inflections    
lancaster = LancasterStemmer() # gives root words that might NOT be part of a language, recent and more drastic

for index, sent in enumerate(sentence):
    x = word_tokenize(sent)                # split into words
    x = [w.lower() for w in x]          # lower case
    x = [re_punc.sub('', w) for w in x] # remove punctuation
    x = [w for w in x if w.isalpha()]   # remove tokens that are not alphanumeric
    x = [w for w in x if not w in stopwords.words('english')] # remove stop words
    # apply stemming
    #x = [lancaster.stem(w) for w in x]
    sentence[index] = x


# # Build Model

# In[32]:


EMBEDDING_DIM = 30

start= timer()

model = Word2Vec(size = EMBEDDING_DIM, window=2, min_count=1, compute_loss=True)
model.build_vocab(sentence)
model.train(sentence, total_examples= model.corpus_count, epochs= 50)

end = timer()

print("Time taken:", end-start)


# In[33]:


# See similarity b/w words
model.wv.most_similar(positive=['good'], topn = 3)


# # Get Embeddings

# In[34]:


#Important Parameters
VOCAB_SIZE    = len(model.wv.vocab)
embedding     = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

# create a stack vectors for all words embeddings.
for i,word in enumerate(model.wv.index2word[:VOCAB_SIZE]):
        embedding[i] = model[word]


# # T-SNE

# In[35]:


tsne        = TSNE(2)
tsne_data   = tsne.fit_transform(embedding)

# graph
colors = np.array(['#009AFF', '#FF00AB'])

fig, ax = plt.subplots(1, 1, figsize=(15, 12))
fig.suptitle('TSNE: Word Embeddings')

x_coords = embedding[:, 0]
y_coords = embedding[:, 1]
    # display scatter plot
ax.scatter(x_coords, y_coords)

for label, x, y in zip(model.wv.vocab, x_coords, y_coords):
    ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

ax.title.set_text("Z-Code TSNE")
plt.show()


# In[36]:


for i in model.wv.vocab:
    print(i)


# In[ ]:





# In[ ]:




