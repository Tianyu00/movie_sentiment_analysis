#### import data (this takes a while ~ 3min)
import pandas as pd
import os
basepath='/Users/tz/Desktop/aclImdb'
labels={'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file),
                     'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],
                          ignore_index=True)
df.columns = ['review', 'sentiment']
# df.shape: (50000, 2)



#### cleaning text data
from nltk.corpus import stopwords
import re
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                          text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized



#### SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
import random
vect = HashingVectorizer(decode_error='ignore',
                        n_features = 2**21,
                        preprocessor=None,
                        tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1)

def get_minibatch(df, size):
    df_sample = df.sample(n=size, replace=True)
    return df_sample['review'], df_sample['sentiment']

classes = np.array([0,1])
for _ in range(45):
    X_train, y_train = get_minibatch(df, size=1000)
    if X_train is None:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes = classes)



#### save model
import pickle
import os
dest = os.path.join('/Users/tz','Desktop','web_part','pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop,
           open(os.path.join(dest,'stopwords.pkl'),'wb'),
           protocol=4)
pickle.dump(clf,
           open(os.path.join(dest,'classifier.pkl'),'wb'),
           protocol=4)
