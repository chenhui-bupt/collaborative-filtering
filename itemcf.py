#-*- coding: utf-8 -*-
'''
Created on 2017-12-23

@author: chenhui
'''
import sys, random, math
from operator import itemgetter
from cf import CF

random.seed(0)


class ItemBasedCF(CF):
    ''' TopN recommendation - ItemBasedCF '''
    def __init__(self):
        super().__init__()
        self.n_sim_item = 20
        self.n_rec_item = 10
        print('Similar item number = %d' % self.n_sim_item)
        print('Recommended item number = %d' % self.n_rec_item)


    def calc_item_sim(self):
        ''' calculate item similarity matrix '''
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

        # count co-rated users between items
        itemsim_mat = self.item_sim_mat
        print('building co-rated users matrix...')

        for user, items in self.trainset.items():
            for m1 in items:
                for m2 in items:
                    if m1 == m2: continue
                    itemsim_mat.setdefault(m1,{})
                    itemsim_mat[m1].setdefault(m2,0)
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ')

        # calculate similarity matrix 
        print('calculating item similarity matrix...')
        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_items in itemsim_mat.items():
            for m2, count in related_items.items():
                itemsim_mat[m1][m2] = count / math.sqrt(
                        self.item_popular[m1] * self.item_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating item similarity factor(%d)' % simfactor_count)

        print('calculate item similarity matrix(similarity factor) succ')
        print('Total similarity factor number = %d' %simfactor_count)


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


