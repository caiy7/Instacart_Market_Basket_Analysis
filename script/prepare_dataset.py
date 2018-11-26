#!/usr/bin/python

import pandas as pd
import numpy as np
from collections import OrderedDict, Counter
import pickle

def merge_order_product_df(order_product, prior_reorder_summary, order_info):
    '''
    Add detailed order product info to each order by mergering the order_product with orders. 
    Argument: 

    '''
    selected_user_id = order_info[order_info['order_id'].isin(order_product['order_id'])]['user_id']
    prior_valid_user = prior_reorder_summary[prior_reorder_summary['user_id'].isin(selected_user_id)]
    order_details = pd.merge(order_product, order_info.drop(columns='eval_set'), on='order_id', how='left')
    df = pd.merge(prior_valid_user, 
                    order_details[['user_id', 'product_id', 'reordered']], 
                    on=['user_id', 'product_id'], how='left')
    df['reordered'].fillna(0,inplace=True)
    return df, order_details


def merge_user_product_feature(df, user_product_feature, days_since_first_order):
    df = pd.merge(df, user_product_feature.drop(columns='order_number'), 
                    on=['user_id', 'product_id'], how='left')
    # * user_product_percent_order_containing_product 
    df['user_product_percent_order_containing_product'] = ((df['user_product_reordered_past'] + 1)
                                                    / df['user_total_order'])
    df = pd.merge(df, days_since_first_order, on='user_id', how='left')
    df['user_product_ave_day_per_product_order'] = df['days_since_first_order']/(df['user_product_reordered_past']+1)
    df.drop(columns='days_since_first_order', inplace=True)
    return df


def process_current_order(row):
    # current_order_since_last_product_purchase_vs_average_purchase_gap
    days_since_last_purchase = row['user_product_days_before_last_product_order'] \
    + row['current_days_since_prior_order']
    # take care of 0
    if row['user_product_ave_day_per_product_order'] == 0:
        row['user_product_ave_day_per_product_order']+=0.5
    if days_since_last_purchase == 0:
        days_since_last_purchase+=0.5
    purchase_days_ratio = days_since_last_purchase\
    /row['user_product_ave_day_per_product_order']
    #deviation on the gaps from current compared to the average of the user
    if row['user_days_since_last_order_std'] == 0:
        gaps_deviation = None
    else:
        gaps_deviation = (row['current_days_since_prior_order'] - 
                          row['user_days_since_last_order_mean'])/row['user_days_since_last_order_std']
    # whether the user has placed an order of the same product that day   
    if row['current_days_since_prior_order'] == 0 and \
    row['user_total_order'] == row['user_product_most_recent_order']:
        same_day = 1
    else:
        same_day = 0
    return days_since_last_purchase, purchase_days_ratio, gaps_deviation, same_day
    

def current_order_specific_features(df, order_details):
    current_order_specific = order_details.groupby('user_id', as_index=False).agg(OrderedDict([('days_since_prior_order', 'first'), ('order_hour_of_day', 'first')]))
    df = pd.merge(df, current_order_specific, on='user_id', how='left')
    df.rename(columns={'days_since_prior_order':'current_days_since_prior_order'},inplace=True)
    (df['current_order_days_since_last_product_purchase'], 
        df['current_order_since_last_product_purchase_vs_average_purchase_gap'],
        df['current_order_gap_deviation'],
        df['current_same_day_order']) = zip(*df.apply(process_current_order, axis=1))
    df['current_order_gap_deviation'].fillna(99999,inplace=True)
    df.drop(columns=['user_product_most_recent_order', 'user_product_days_before_last_product_order', 'order_hour_of_day', 'user_days_since_last_order_std'], inplace=True)

    return df


def feature_engineering(train_order_product, test_order_product, prior_order_product):
    #make df
    order_info = pd.read_csv('../raw_data/orders.csv')
    prior_order_details = pd.merge(prior_order_product, order_info.drop(columns='eval_set'), on='order_id', how='left')
    prior_reorder_summary = prior_order_details.groupby(['user_id','product_id'], as_index=False)['reordered'].sum()
    prior_reorder_summary.rename({'reordered':'user_product_reordered_past'}, axis=1, inplace=True)

    # Get all the product the user ordered in the past and predict if he will reorder in his recent order. 
    # Assign the reordered information (0 or 1) to the prior order (groupby user and product id)
    train_df, train_order_details = merge_order_product_df(train_order_product, prior_reorder_summary, order_info)
    test_df, test_order_details = merge_order_product_df(test_order_product, prior_reorder_summary, order_info)
    
    del train_order_product, test_order_product, prior_reorder_summary, order_info

    # Add feature: 
    # Also add user_total_order, user_cart_size, user_total_product,user_ave_days_since_last_order
    user_feature = (prior_order_details.groupby(['user_id', 'order_id'], as_index=False)
                .agg(OrderedDict([('product_id', 'count'), ('days_since_prior_order', 'first')])))
    user_feature.rename(columns={'product_id': 'total_product'}, inplace=True)
    user_feature = (user_feature.groupby(['user_id'], as_index=False)
                .agg(OrderedDict([('order_id','count'), ('total_product', ['mean', 'sum']), ('days_since_prior_order', ['mean', 'std'])])))
    user_feature_colname = ['user_id', 'user_total_order', 'user_cart_size', 'user_total_product', 'user_days_since_last_order_mean', 
                            'user_days_since_last_order_std']
    user_feature.columns = user_feature_colname

    # Add number of product that the user ordered only once before - user_total_uniq_product
    # and percentage of total product that are unique - total_uniq_over_total_product
    user_uniq_product = prior_order_details.groupby('user_id')['product_id'].nunique()
    user_feature = pd.merge(user_feature, user_uniq_product.to_frame(), left_on ='user_id', 
                            right_index=True, how='left')
    user_feature.rename(columns={'product_id': 'user_total_uniq_product'}, inplace=True)
    user_feature['user_uniq_prod_over_total_prod'] = \
    user_feature['user_total_uniq_product']/user_feature['user_total_product']
    del user_uniq_product

    #merge user feature into the dataframe
    train_df = pd.merge(train_df, user_feature, on='user_id', how='left')
    test_df = pd.merge(test_df, user_feature, on='user_id', how='left')
    print('after added user feature: ')
    print('train_df: ', train_df.shape)
    print('test_df: ', test_df.shape)
    print('User features added.\nAdding product features...')

    # Add product specific features:
    # *Total counts of the product were ordered in the past over all users - product_total_orders
    # *Averge product add to cart order - product_avg_add_to_cart_order
    # *percentage of orders that contains the product - product_percent_order_containing_product
    prod_feature_names = ['product_total_orders', 'product_avg_add_to_cart_order']
    prod_feature = prior_order_details.groupby('product_id', as_index=False).agg(OrderedDict([('order_id','count'), 
                                                                        ('add_to_cart_order', 'mean')]))
    prod_feature.columns = ['product_id'] + prod_feature_names
    prod_feature['product_percent_order_containing_product'] = prod_feature['product_total_orders']/prior_order_details['order_id'].nunique()

    #remember to save prod_feature for preparing test
    train_df = pd.merge(train_df, prod_feature, on='product_id', how='left')
    test_df = pd.merge(test_df, prod_feature, on='product_id', how='left')
    print('after added product feature: ')
    print('train_df: ', train_df.shape)
    print('test_df: ', test_df.shape)
    print('Product features added.\nAdding user_product features...')

    # Add user-product feature
    # * user-product_ave_add_to_cart_order
    # * user_product_ave_add_to_cart_order_scale(average of add to cart over scaled within each user)

    prior_order_details['user_product_add_to_cart_order_scale']=(prior_order_details.groupby('order_id')['add_to_cart_order']
                                                               .transform(lambda x: x/max(x)))
    user_product_feature = (prior_order_details.groupby(['user_id', 'product_id'], as_index=False)
                       .agg(OrderedDict([('user_product_add_to_cart_order_scale', 'mean'),('days_since_prior_order', 'mean'),
                                        ('order_number', 'max')])))
    user_product_feature.rename(columns={'user_product_add_to_cart_order_scale':'user_product_ave_add_to_cart_order_scale',
                                    'days_since_prior_order':'user_product_ave_days_since_prior_order',
                                    'order_number':'user_product_most_recent_order'}, inplace=True)
    # * user_product_ordered_last_n_order - how many orders ago is the last time the user ordered that product
    print('Adding feature - user_product_ordered_last_n_order')
    user_product_feature = pd.merge(user_product_feature, user_feature[['user_id', 'user_total_order']], 
                on='user_id', how='left')
    user_product_feature['user_product_ordered_last_n_order'] = (user_product_feature['user_total_order']
                                                - user_product_feature['user_product_most_recent_order'])
    user_product_feature.drop(columns='user_total_order', inplace=True)
    # * user_product_days_since_last_product_order.
    # The prior days of the first order is null. fill NA with ffill with addtional 7 days (a week). 
    # The rational to use short period versus long period for the cases that the user is new to Instacart.

    print('Adding feature - user_product_days_since_last_product_order')
    df_temp=(prior_order_details.groupby(['user_id', 'order_number'], as_index=False)
    .agg(OrderedDict([('order_id', 'first'), ('days_since_prior_order', 'first')]))
    .sort_values(['user_id', 'order_number'], ascending=[True, False]))
    df_temp['cumulative_days_since_nth_order'] = (df_temp.groupby('user_id', as_index=False)['days_since_prior_order']
                                                .cumsum())
    # The prior days of the first order is null. fill NA with ffill with addtional 7 days (a week). 
    # The rational to use short period versus long period for the cases that the user is new to Instacart.
    a = df_temp['cumulative_days_since_nth_order'].fillna(method='ffill').copy()
    a[df_temp['cumulative_days_since_nth_order'].isnull()] +=7
    df_temp['cumulative_days_since_nth_order']=a.copy()
    del a
    user_product_feature = pd.merge(user_product_feature, df_temp.drop(columns=['order_id', 'days_since_prior_order']), 
                        left_on=['user_id', 'user_product_most_recent_order'], right_on = ['user_id', 'order_number'],
                        how='left')
    del df_temp
    user_product_feature.rename(columns={'cumulative_days_since_nth_order'
                                     :'user_product_days_before_last_product_order'}, inplace=True)
    # user_product_ave_days_since_prior_order have NAs. 
    # Those are the product the customers ordered only once (in the very first order. 
    # Fill NA with user_product_cumulative_days_since_nth_order for the produce (maximum period that the user didn't order the product in this data set.
    user_product_feature['user_product_ave_days_since_prior_order'].fillna(user_product_feature['user_product_days_before_last_product_order'], inplace=True)  
    days_since_first_order = (prior_order_details.groupby(['user_id','order_id'], as_index=False)
                            .agg(OrderedDict([('days_since_prior_order', 'first')]))
                            .groupby('user_id', as_index=False)['days_since_prior_order'].sum())
    days_since_first_order.rename(columns={'days_since_prior_order':'days_since_first_order'}, inplace=True)
    
    train_df = merge_user_product_feature(train_df, user_product_feature, days_since_first_order)
    test_df = merge_user_product_feature(test_df, user_product_feature, days_since_first_order)
    
    del days_since_first_order
    del user_product_feature

    print('Adding feature - user_product_ave_day_between_product_order_versus_current_day_since_prior')
    
    train_df = current_order_specific_features(train_df, train_order_details)
    test_df = current_order_specific_features(test_df, test_order_details)

    # add binned categories
    train_df, dept_map, aisle_map = add_binned_category(train_df)
    test_df, _, _= add_binned_category(test_df, test=True, dept_map=dept_map, aisle_map=aisle_map)

    with open('../data/bin_maps.pkl', 'wb') as f:
        pickle.dump((dept_map, aisle_map), f)

    return train_df, test_df

def add_binned_category(df, test=False, dept_map=None, aisle_map=None):

    product_df=pd.read_csv('../raw_data/products.csv')
    df=pd.merge(df, product_df, on='product_id', how='left')
    df.drop(columns='product_name', inplace=True)

    # creat bin by department
    print('Binnig department...')
    if not test:
        no_order_ratio = df.loc[df['reordered']==0,'department_id'].value_counts().sort_index()/\
        df.loc[df['reordered']==0,'department_id'].count()
        re_order_ratio = df.loc[df['reordered']==1,'department_id'].value_counts().sort_index()/\
        df.loc[df['reordered']==1,'department_id'].count()

        dept_freq  = pd.DataFrame(no_order_ratio)
        dept_freq.columns=['no_reorder']
        dept_freq = pd.merge(dept_freq, re_order_ratio.to_frame(), left_index=True, right_index=True, how='outer')
        dept_freq.rename(columns={'department_id':'reordered'}, inplace =True)
        dept_freq['diff'] = dept_freq['reordered'] - dept_freq['no_reorder']
        dept_freq.sort_values('diff', ascending=False, inplace=True)

        dept_freq['dept_bin']=0
        dept_freq.loc[(dept_freq['diff']>=0.02), 'dept_bin'] = 2
        dept_freq.loc[(dept_freq['diff']<0.02) & (dept_freq['diff']>= 0 ),  'dept_bin'] = 1
        dept_freq.loc[(dept_freq['diff']<0) & (dept_freq['diff']>= -0.01 ),  'dept_bin'] = -1
        dept_freq.loc[(dept_freq['diff']<-0.01) & (dept_freq['diff']>= -0.02 ),  'dept_bin'] = -2
        dept_freq.loc[(dept_freq['diff']<-0.02),  'dept_bin'] = -3
        dept_map = dept_freq['dept_bin']
    df['product_dept_bin'] = df['department_id'].map(dept_map)
    df.drop(columns='department_id', inplace=True)

    # create bin by aisle
    print('Binnig aisle...')
    if not test:
        no_order_ratio = df.loc[df['reordered']==0,'aisle_id'].value_counts().sort_index()/\
        df.loc[df['reordered']==0,'aisle_id'].count()
        re_order_ratio = df.loc[df['reordered']==1,'aisle_id'].value_counts().sort_index()/\
        df.loc[df['reordered']==1,'aisle_id'].count()

        aisle_freq  = pd.DataFrame(no_order_ratio)
        aisle_freq.columns=['no_reorder']
        aisle_freq = pd.merge(aisle_freq, re_order_ratio.to_frame(), left_index=True, right_index=True, how='outer')
        aisle_freq.fillna(0, inplace=True)
        aisle_freq.rename(columns={'aisle_id':'reordered'}, inplace =True)
        aisle_freq['diff'] = aisle_freq['reordered'] - aisle_freq['no_reorder']
        aisle_freq['product_aisle_bin'] = 0
        aisle_freq.loc[aisle_freq['diff']>=0.001, 'product_aisle_bin'] = 2
        aisle_freq.loc[aisle_freq['diff']<=-0.001, 'product_aisle_bin'] = -2
        aisle_freq.loc[(aisle_freq['diff']<0.001) & (aisle_freq['diff']>=0), 'product_aisle_bin'] = 1
        aisle_freq.loc[(aisle_freq['diff']>-0.001) & (aisle_freq['diff']<0), 'product_aisle_bin'] = -1
        aisle_map = aisle_freq['product_aisle_bin']
    df['product_aisle_bin'] = df['aisle_id'].map(aisle_map)
    df.drop(columns='aisle_id', inplace=True)

    return df, dept_map, aisle_map


if __name__ == '__main__':
    train_order_product = pd.read_csv('../data/train_order_products.csv')
    prior_order_product = pd.read_csv('../raw_data/order_products__prior.csv')
    test_order_product = pd.read_csv('../data/test_order_products.csv')

    train_df, test_df = feature_engineering(train_order_product, test_order_product, prior_order_product)
    print('train_df shape: ',  train_df.shape)
    print('test df shape: ', test_df.shape)
    with open('../data/final_dataset.pkl', 'wb') as f:
        pickle.dump((train_df, test_df), f)





