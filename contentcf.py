#-*- coding: utf-8 -*-
'''
Created on 2017-12-23

@author: chenhui
'''
import sys, random, math
from operator import itemgetter
from cf import CF
from contentsim import contentsim

random.seed(0)


class ContentBasedCF(CF):
    ''' TopN recommendation - ContentBasedCF '''
    def __init__(self):
        super().__init__()
        self.n_sim_item = 10
        self.n_rec_item = 10
        print('Similar item number = %d' % self.n_sim_item)
        print('Recommended item number = %d' % self.n_rec_item)


    def calc_item_sim(self, calc_similarity): # calc_similarity is a function parameter
        print('counting items number and popularity...')

        for user, items in self.trainset.items():
            for item in items:
                # count item popularity 
                if item not in self.item_popular:
                    self.item_popular[item] = 0
                self.item_popular[item] += 1

        print('count items number and popularity succ')
        # save the total number of items
        self.item_count = len(self.item_popular)
        print('total item number = %d' % self.item_count)

        ''' calculate item similarity matrix '''
        print('calculating item similarity matrix...')
        self.item_sim_mat = calc_similarity()
        print('calculate item similarity matrix(similarity factor) succ')


    def recommend(self, user):
        ''' Find K similar items and recommend N items. '''
        K = self.n_sim_item
        N = self.n_rec_item
        rank = {}
        watched_items = self.trainset[user]

        for item, rating in watched_items.items():
            for related_item, w in sorted(self.item_sim_mat[item].items(),
                    key=itemgetter(1), reverse=True)[:K]:
                if related_item in watched_items:
                    continue
                rank.setdefault(related_item, 0)
                rank[related_item] += w * rating
        # return the N best items
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

