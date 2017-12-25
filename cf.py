import sys, random, math
from operator import itemgetter
import codecs
import numpy as np

random.seed(0)


class CF():
    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_user = 20
        self.n_sim_item = 20
        self.n_rec_item = 10

        self.item_sim_mat = {}
        self.user_sim_mat = {}
        self.item_popular = {}
        self.item_count = 0


    @staticmethod
    def loadfile(filename, header=False):
        ''' load a file, return a generator. '''
        fp = codecs.open(filename, 'r', encoding='gbk')
        for i,line in enumerate(fp):
            if header==False and i==0:
                continue
            yield line.strip('\r\n')
            if i%100000 == 0:
                print('loading %s(%s)' % (filename, i))
        fp.close()
        print('load %s succ' % filename)


    def generate_dataset(self, filename, separate=',', index=range(3), pivot=0.7):
        ''' load rating data and split it to training set and test set '''
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(filename):
            user, item, rating = np.array(line.split(separate))[index]
            # split the data by pivot
            if (random.random() < pivot):
                self.trainset.setdefault(user,{})
                self.trainset[user].setdefault(item, 0)
                self.trainset[user][item] += float(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user,{})
                self.testset[user].setdefault(item, 0)
                self.testset[user][item] += float(rating)
                testset_len += 1

        print('split training set and test set succ')
        print('train set = %s' % trainset_len)
        print('test set = %s' % testset_len)


    def recommend(self, user):
        return


    def evaluate(self):
        ''' return precision, recall, coverage and popularity '''
        print('Evaluation start...')

        N = self.n_rec_item
        #  varables for precision and recall 
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_items = set()
        # varables for popularity
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('recommended for $%d users' % i)
            test_items = self.testset.get(user, {})
            rec_items = self.recommend(user)
            for item, w in rec_items:
                if item in test_items:
                    hit += 1
                all_rec_items.add(item)
                popular_sum += math.log(1 + self.item_popular[item])
            rec_count += N
            test_count += len(test_items)

        precision = hit / (1.0*rec_count)
        recall = hit / (1.0*test_count)
        coverage = len(all_rec_items) / (1.0*self.item_count)
        popularity = popular_sum / (1.0*rec_count)

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' % \
                (precision, recall, coverage, popularity))

        