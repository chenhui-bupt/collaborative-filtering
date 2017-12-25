from usercf import UserBasedCF
from itemcf import ItemBasedCF
from contentcf import ContentBasedCF
from contentsim import contentsim
import datetime

if __name__ == '__main__':
    train_ratingfile = '../datasets/nfp/data0117.csv'
    test_ratingfile = '../datasets/nfp/data0118.csv'

    ratingfile = '../datasets/ml-latest-small/ratings.csv'
    starttime = datetime.datetime.now()
    print("usercf...")
    usercf = UserBasedCF()
    usercf.generate_dataset(train_ratingfile,index=[1,3,5], pivot=1.0) # user,item,rating
    usercf.generate_dataset(test_ratingfile,index=[1,3,5], pivot=0.0) # user,item,rating
    usercf.calc_user_sim()
    usercf.evaluate()
    endtime = datetime.datetime.now()
    print("usercf has spend %s time" % str(endtime-starttime)) 

    starttime = datetime.datetime.now()
    print('itemcf...')
    itemcf = ItemBasedCF()
    itemcf.generate_dataset(train_ratingfile, index=[1,3,5], pivot=1.0)
    itemcf.generate_dataset(test_ratingfile, index=[1,3,5], pivot=0.0)
    itemcf.calc_item_sim()
    itemcf.evaluate()
    endtime = datetime.datetime.now()
    print("itemcf has spend %s time" % str(endtime-starttime)) 

    starttime = datetime.datetime.now()
    print("contetncf...")
    contentcf = ContentBasedCF()
    contentcf.generate_dataset(train_ratingfile, index=[1,3,5], pivot=1.0)
    contentcf.generate_dataset(test_ratingfile, index=[1,3,5], pivot=0.0)
    
    contentcf.calc_item_sim(contentsim.calc_tfidf_similarity)
    contentcf.evaluate()

    contentcf.calc_item_sim(contentsim.calc_lda_similarity)
    contentcf.evaluate()

    contentcf.calc_item_sim(contentsim.calc_lsi_similarity)
    contentcf.evaluate()
    
    endtime = datetime.datetime.now()
    print("contentcf has spend %s time" % str(endtime-starttime)) 



# 结果：
# precision=0.0665	recall=0.0888	coverage=0.9818	popularity=3.5915
# precision=0.1244	recall=0.1662	coverage=0.7455	popularity=8.5361
# precision=0.0573	recall=0.0766	coverage=1.0000	popularity=6.7215


# 内容过滤：item_sim =20
# precision=0.0573	recall=0.0766	coverage=1.0000	popularity=6.7215
# precision=0.0260	recall=0.0348	coverage=1.0000	popularity=6.1330
# precision=0.0533	recall=0.0712	coverage=0.9455	popularity=7.5160

# 内容过滤：item_sim =10
# precision=0.0491	recall=0.0656	coverage=1.0000	popularity=6.5315
# precision=0.0365	recall=0.0488	coverage=0.9818	popularity=6.8952
# precision=0.0505	recall=0.0675	coverage=0.9455	popularity=7.0904

