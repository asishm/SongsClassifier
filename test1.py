import pandas as pd
import os.path

curdir = r"C:\Users\Asish\Documents\Courses\Python Based Data Analytics\Final Project"
print(curdir)

#for j in range(0,10):
#    year = ['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
#    dHome = "/Users/HimanshuBharara/Documents/CU-Sem2/IEORE4571/Projects/Lyrics/"+year[j]+"/"
#    print(dHome)

dHome = os.path.join(curdir,"Lyrics/2006/")
print(dHome)
lyrics_array = []
print(lyrics_array)
for i in range(0,100):
        try:
            name=dHome+str(i)+"."+"txt"
            file=open(name,'r+')
            lyrics_array.append(file.read())
            #globals()['string%s' % i] = file.read()
            file.close()
        except ValueError:
            print(i)

dHome = os.path.join(curdir, "Lyrics/2006/")
name= os.path.join(curdir, "Lyrics/2006/74.txt")
file=open(name,'r+')

from collections import Counter
import re

df = {}
tf = {}

for doc in lyrics_array:
    for word in doc.split():
        k = word
        word = word.lower()
        word = re.sub(r'[\'\"\:\;\/\?\.\>\,\<\]\}\[{=\-\_\)\(\`\~\d\!]', '', word)
        word.strip()
        tf[word] = tf.get(word, 0) + 1
        df[word] = df.get(word, []) + [k]

vocabulary = list(df.keys())

##
##
##import string #allows for format()
##    
##def build_lexicon(corpus):
##    lexicon = set()
##    for doc in corpus:
##        lexicon.update([word for word in doc.split()])
##    return lexicon
##
##def tf(term, document):
##  return freq(term, document)
##
##def freq(term, document):
##  return document.split().count(term)
##
##vocabulary = build_lexicon(lyrics_array)
##
##doc_term_matrix = []
###print ('Our vocabulary vector is [' + ', '.join(list(vocabulary)) + ']')
##for doc in lyrics_array:
##    #print ('The doc is "' + doc + '"')
##    tf_vector = [tf(word, doc) for word in vocabulary]
##    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
##    #print ('The tf vector for Document %d is [%s]' % ((lyrics_array.index(doc)+1), tf_vector_string))
##    doc_term_matrix.append(tf_vector)
##    
##    # here's a test: why did I wrap mydoclist.index(doc)+1 in parens?  it returns an int...
##    # try it!  type(mydoclist.index(doc) + 1)
##
##print ("All combined, here is our master document term matrix:")
###print (doc_term_matrix)
##
##import math
##import numpy as np
##
##def l2_normalizer(vec):
##    denom = np.sum([el**2 for el in vec])
##    if denom == 0:
##        return [(0) for el in vec]
##    else:
##        return [(el / math.sqrt(denom)) for el in vec]
##    
##doc_term_matrix_l2 = []
##for vec in doc_term_matrix:
##    doc_term_matrix_l2.append(l2_normalizer(vec))
##
##print ('A regular old document term matrix: ')
##print (np.matrix(doc_term_matrix))
##print ('\nA document term matrix with row-wise L2 norms of 1:')
##print (np.matrix(doc_term_matrix_l2))
##
##def numDocsContaining(word, doclist):
##    doccount = 0
##    for doc in doclist:
##        if freq(word, doc) > 0:
##            doccount +=1
##    return doccount 
##
##def idf(word, doclist):
##    n_samples = len(doclist)
##    df = numDocsContaining(word, doclist)
##    return np.log(n_samples / 1+df)
##
##my_idf_vector = [idf(word, lyrics_array) for word in vocabulary]
##
##print(len(vocabulary))
##
##import numpy as np
##
##def build_idf_matrix(idf_vector):
##    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
##    np.fill_diagonal(idf_mat, idf_vector)
##    return idf_mat
##
##my_idf_matrix = build_idf_matrix(my_idf_vector)
##
##doc_term_matrix_tfidf = []
##dHome = curdir
###performing tf-idf matrix multiplication
##for tf_vector in doc_term_matrix:
##    doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))
##
###normalizing
##doc_term_matrix_tfidf_l2 = []
##for tf_vector in doc_term_matrix_tfidf:
##    doc_term_matrix_tfidf_l2.append(l2_normalizer(tf_vector))
##                                    
###print vocabulary
##i=2006
##print(np.matrix(doc_term_matrix_tfidf_l2)) # np.matrix() just to make it easier to look at
##dic_t = dHome+str(i)+"_"+"dict"+".txt"
##vec=dHome+str(i)+"_"+"vector"+"."+"txt"
##
##from nltk.corpus import wordnet
##
##list1 = vocabulary
##list2 = ['anger', 'surprise', 'joy', 'sadness', 'love', 'fear']
##
##list3 = []
##
##for i,word1 in enumerate(list1):
##    k = []
##    for word2 in list2:
##        wordFromList1 = wordnet.synsets(word1)
##        wordFromList2 = wordnet.synsets(word2)
##        if wordFromList1 and wordFromList2: #Thanks to @alexis' note
##            s = wordFromList1[0].wup_similarity(wordFromList2[0])
##            k.append(s)
##    list3.append(k)
##    print(word1, k)
##    if i == 10:
##        break
##b = np.array(list3)
##print(b.shape)
