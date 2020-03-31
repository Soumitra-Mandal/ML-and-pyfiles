

import nltk
import textblob
from textblob import TextBlob

data=TextBlob("Hello Everyone!hope you are enjoying the day.")
data.translate(to="es")
data.translate(to="bn")
data=TextBlob("The orange is a bad fruit")
data.sentiment
data=TextBlob("Ii do have a car.")
data.correct()

#tokenize:divide the sentences into chunks
data="Science revolves around studying observations made of the natural world around us, and engaging in experimentation. Scientific studies are based on conducting experiments in the real world, and in laboratories. Experiments in laboratories are usually conducted under controlled conditions that may not be similar to the real-world conditions. Such, experiments, however are useful in understanding phenomena that occur in the real world."

from nltk import sent_tokenize
sent_tokenize(data)

from nltk import word_tokenize
word_tokenize(data)

from nltk.stem import PorterStemmer
ps=PorterStemmer()
ps.stem("cars")
ps.stem("knives")

ps.stem("observation")

from nltk import stem
wd=stem.WordNetLemmatizer()
wd.lemmatize("knives")
wd.lemmatize("observing")
wd.lemmatize("observing","v")
import string
import nltk
from nltk.corpus import stopwords
#first 20 stopwords in English corpus
len(stopwords.words("English")[0:])
test_sentence="this is my first string test. wow! we are doing good."
no_punctuation=[char for char in test_sentence 
                if char not in string.punctuation]

print(no_punctuation)

no_punctuation="".join(no_punctuation)
print(no_punctuation)
#remove stopwords
clnsnt=[word for word in no_punctuation.split() if word.lower() not in stopwords.words("english")]
print(clnsnt)
import os
import nltk

import nltk.corpus
print(os.listdir(nltk.data.find("corpora")))
nltk.corpus.gutenberg.fileids()

milton=nltk.corpus.gutenberg.words('milton-paradise.txt')
 
AI="""machine learning is a part of artificial intelligence. machine learning is widely used. Artificial intelligence is incomplete without machine learning."""
type(AI)

AI_token=word_tokenize(AI)
AI_token

from nltk.probability import FreqDist
fdist=FreqDist()
for word in AI_token:
    fdist[word.lower()]+=1
fdist
fdist_top10=fdist.most_common(10)
fdist


