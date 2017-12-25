# -*- coding: utf-8 -*-

import gensim
from gensim import corpora, models, similarities
import sys
import codecs
import jieba
import codecs


f = codecs.open('contentsim/stopwords.txt', 'r', encoding='utf-8')
stop_words = ['的',',','，','.','。','、']
for each in f.readlines():
    stop_words.append(each.replace('\n','').replace('‘',''))
f.close()

applist = []
raw_documents = []
with open('contentsim/app_alias.txt', 'r') as f:
    contents = f.readlines()
    for content in contents:
        content = content.split('|')
        applist.append(content[0])
        raw_documents.append(content[2])


corpora_documents = []  
for item_text in raw_documents:  
    item_str = jieba.cut(item_text)
    item = [word for word in item_str if word not in stop_words]
    corpora_documents.append(item)  

  
dictionary = corpora.Dictionary(corpora_documents)  
corpus = [dictionary.doc2bow(text) for text in corpora_documents]  
tfidf = models.TfidfModel(corpus)  
corpus_tfidf = tfidf[corpus]


num_features=len(dictionary.keys()) 


def calc_similarity(corpus_vec):
    item_sim_mat = {}
    similarity = similarities.Similarity('Similarity-tfidf-index', corpus_vec, num_features=num_features)
    for i, sim in enumerate(similarity):
        item = applist[i]
        item_sim_mat[item]=dict(zip(applist, sim))
        item_sim_mat[item].pop(item)
    return item_sim_mat


def calc_tfidf_similarity():
    return calc_similarity(corpus_tfidf)

def calc_lda_similarity():
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, update_every=0, passes=1)  
    corpus_lda = lda[corpus_tfidf]
    return calc_similarity(corpus_lda)

def calc_lsi_similarity():
    lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=10)  
    corpus_lsi = lsi[corpus_tfidf]
    return calc_similarity(corpus_lsi)
            

