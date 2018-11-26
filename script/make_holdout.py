#!/usr/bin/python

'''take the information of the order to be predicted (20,000) out of the order_products_trains.csv file 
'''

import pandas as pd
import numpy as np

def take_holdout():
    recent_order_products = pd.read_csv('../raw_data/order_products__train.csv')
    order_details = pd.read_csv('../raw_data/orders.csv')
    prior_order_products = pd.read_csv('../raw_data/order_products__prior.csv')
    #remove the original Kaggle test set
    print('order_detail dataframe shape: ', order_details.shape)
    order_details = order_details[~(order_details['eval_set']=='test')]
    print('order_detail dataframe shape after remove Kaggle original test set: ', order_details.shape)

    #randomly select 20,000 order in the recent_orders as holdout set and 6000 orders as subset for model selection 
    recent_order_ids = recent_order_products['order_id'].unique()
    subset_with_holdout_ids = np.random.choice(recent_order_ids, size = 26000, replace=False)
    subset_order_ids = np.random.choice(subset_with_holdout_ids, size = 6000, replace=False)
    holdout_order_ids = [x for x in subset_with_holdout_ids if x not in subset_order_ids]
    test_order_products = recent_order_products[recent_order_products['order_id'].isin(holdout_order_ids)]
    train_order_products = recent_order_products[~recent_order_products['order_id'].isin(holdout_order_ids)]
    subset_order_products = recent_order_products[recent_order_products['order_id'].isin(subset_order_ids)]
    print('train_order_products dataframe shape: ', train_order_products.shape)
    print('test_order_products dataframe shape: ', test_order_products.shape)
    print('subset_order_products dataframe shape: ', subset_order_products.shape)

    # take subset of prior_order, which should include all the orders from the users in the subset and orders from additional users in the prior orders.
    user_ids = order_details['user_id'].unique()
    subset_user_ids = order_details[order_details['order_id'].isin(subset_order_ids)]['user_id']
    additional_user_id = np.random.choice(user_ids, 6000, replace=False)
    sub_all_user_ids = list(set(additional_user_id) | set(subset_user_ids))
    print('number of users in sub_prior_order_product dataset: ', len(sub_all_user_ids))
    subset_prior_order_ids = order_details[order_details['user_id'].isin(sub_all_user_ids)]['order_id']
    subset_prior_order_products = prior_order_products[prior_order_products['order_id'].isin(subset_prior_order_ids)]
    print('subset_prior_order_products dataframe shape: ', subset_prior_order_products.shape)

    #save train and test datasets
    train_order_products.to_csv('../data/train_order_products.csv', index=False)
    test_order_products.to_csv('../data/test_order_products.csv', index=False)
    subset_order_products.to_csv('../data/subset_order_products.csv', index=False)
    subset_prior_order_products.to_csv('../data/subset_prior_order_products.csv', index=False)

if __name__ == '__main__':
    take_holdout()

