####!# Libraries ####!#

import sys, webbrowser, os, bz2, re, gc
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from keras.preprocessing import text
import nltk

####!# Related sources ####!#

webbrowser.open_new_tab('https:/www.kaggle.com/anshulrai/cudnnlstm-implementation-93-7-accuracy')

####!# Paths for imports ####!#

path = 'C:\\Users\\Marek.Pytka\\AppData\\Roaming\\Python\\Python37\\site-packages\\'
sys.path.insert(0,'C:\\Users\\Marek.Pytka\\AppData\\Roaming\\Python\\Python37\\site-packages\\')
os.environ["PATH"] += os.pathsep + path
sys.path
os.chdir('C:\\Users\\Marek.Pytka\\Desktop\\Inne szkolenia\\Amazon')
pos_tags_defs = pd.read_csv("POS_Tags.csv", encoding = "cp1252")
#### Loading FastTextData

import bz2
train_file = bz2.BZ2File('train.ft.txt.bz2')
train_file_lines = train_file.readlines()

####!# Conversion from RawBinary to parsable strings ####!#
train_file_lines = [x.decode('utf-8') for x in train_file_lines]

####!# Separation to labels / sentences ####!#

train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

####!#  Simple substitution ####!#

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])
    
####!# Replacements for the websites ####!#
    
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

####!# Delete imported objects ####!#
del train_file_lines

####!# Freeing empty memory in Python ####!#

gc.collect()

max_features = 20000
maxlen = 100

####!# Tokenizing using Keras ####!# 
#
#tokenizer = text.Tokenizer(num_words=max_features)
#tokenizer.fit_on_texts(train_sentences)
#tokenized_train = tokenizer.texts_to_sequences(train_sentences)

####!# Embedding ####!#

#EMBEDDING_FILE = 'twitter.txt'
#
#def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
#embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding = "utf-8"))
#
#for o in open(EMBEDDING_FILE, encoding = "utf-8"):
#    print(o)
    
####!# Creating tokens ####!#
    
tokens = [str(t).split() for t in train_sentences[1:10000]]

from nltk.corpus import stopwords
sr= stopwords.words('english')
clean_tokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)

clean_tokens = pd.DataFrame(clean_tokens)

rows = list([i for i in np.arange(0,clean_tokens.shape[0],1)])
columns = list([i for i in np.arange(0,clean_tokens.shape[1],1)])

clean_tokens = np.array(clean_tokens)

import re
regex = re.compile('[^a-zA-Z]')
regex.sub('', 'TP!@$WC')

for i in rows:
    for j in columns:    
    #clean_tokens[i][j] = str(clean_tokens[i][j]).replace(':','')
        clean_tokens[i][j] = regex.sub('', str(clean_tokens[i][j])) 

clean_tokens = clean_tokens.tolist()
final_list = []

stops = ['the', 'and', 'a', 'i', 'to', 'it', 'this'
         ,'is', 'in', 'that', 'you', 'of', 'for', ''
         ,'have', 'be']

from nltk.stem import PorterStemmer
stemmer = PorterStemmer() 

for token in clean_tokens:
    for word in token:
        if (word != None and word != "None" and word not in stops):
            word = word.lower()
            word = stemmer.stem(word) # przejscie z 38 tysiecy do 28 tysiecy
            final_list.append(word)

token_counter = pd.Series(final_list).value_counts()

pos_tags = nltk.pos_tag(list(final_list))
pos_tags = pd.DataFrame(pos_tags)
pos_tags.columns = ['word', 'pos_tag']
pos_tags['pos_tag'].unique()

pos_tag_counter = pd.Series(pos_tags['pos_tag']).value_counts()
### Series of 5 plots with 20 next most popular words ###

for i in range(1,6):

    cmdstring = 'f' + str(i) + ' = plt.figure(' + str(i) + ')'
    exec(cmdstring)

    start = (i-1) * 20
    end = ((i-1) * 20) + 20 

    plt.plot(token_counter[start:end])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title('Most popular words: rank (' + str(start) + '-' + str(end) + ')')
    print(cmdstring)
    
plt.show()

grouped_pt = pos_tags.groupby(['pos_tag']).size()
grouped_pt = pd.DataFrame({'CC':grouped_pt.index, 'Counter':grouped_pt.values})
grouped_pt["CC"] = grouped_pt["CC"].str.strip()
pos_tags_defs["CC"] = pos_tags_defs["CC"].str.strip()
final_pos_tags = pd.merge(grouped_pt, pos_tags_defs, on = "CC")

import mplcursors
fig, ax = plt.subplots()
ax.bar(x = final_pos_tags["CC"], height = final_pos_tags["Counter"])
ax.set_title("Mouse over a point")
ax.set_xticklabels(labels = final_pos_tags["CC"], rotation=65)
mplcursors.cursor(hover=True)
plt.tight_layout()

