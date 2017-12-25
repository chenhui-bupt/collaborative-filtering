#-*- coding: utf-8 -*-
'''
Created on 2017-12-23

@author: chenhui
'''
import sys, random, math
from operator import itemgetter
from cf import CF


random.seed(0)


class UserBasedCF(CF):
    ''' TopN recommendation - UserBasedCF '''
    def __init__(self):
        super().__init__()
        self.n_sim_user = 20
        self.n_rec_item = 10
        print('Similar user number = %d' % self.n_sim_user)
        print('recommended item number = %d' % self.n_rec_item)


    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        # build inverse table for item-users
        # key=itemID, value=list of userIDs who have seen this item
        print('building item-users inverse table...')
        item2users = dict()

        for user,items in self.trainset.items():
            for item in items:
                # inverse table for item-users
                if item not in item2users:
                    item2users[item] = set()
                item2users[item].add(user)
                # count item popularity at the same time
                if item not in self.item_popular:
                    self.item_popular[item] = 0
                self.item_popular[item] += 1
        print('build item-users inverse table succ')

        # save the total item number, which will be used in evaluation
        self.item_count = len(item2users)
        print('total item number = %d' % self.item_count)

        # count co-rated items between users
        usersim_mat = self.user_sim_mat
        print('building user co-rated items matrix...')

        for item,users in item2users.items():
            for u in users:
                for v in users:
                    if u == v: continue
                    usersim_mat.setdefault(u,{})
                    usersim_mat[u].setdefault(v,0)
                    usersim_mat[u][v] += 1
        print('build user co-rated items matrix succ')

        # calculate similarity matrix 
        print('calculating user similarity matrix...')
        simfactor_count = 0
        PRINT_STEP = 2000000
        for u,related_users in usersim_mat.items():
            for v,count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                        len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calculating user similarity factor(%d)' % simfactor_count)

        print('calculate user similarity matrix(similarity factor) succ')
        print('Total similarity factor number = %d' %simfactor_count)


    def recommend(self, user):
        ''' Find K similar users and recommend N items. '''
        K = self.n_sim_user
        N = self.n_rec_item
        rank = dict()
        watched_items = self.trainset[user]

        # v=similar user, wuv=similarity factor
        for v, wuv in sorted(self.user_sim_mat[user].items(),
                key=itemgetter(1), reverse=True)[0:K]:
            for item in self.trainset[v]:
                if item in watched_items:
                    continue
                # predict the user's "interest" for each item
                rank.setdefault(item,0)
                rank[item] += wuv
        # return the N best items
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]


