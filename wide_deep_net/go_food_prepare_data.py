#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:07:13 2017

@author: anthonyta
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf

wide_input_file = './Final_output_for_wide_part.csv'

iter_csv = pd.read_csv(wide_input_file, iterator=True, chunksize=100000,
						   names=['customer_id',
                                  'recommended_merchant_id',
                                  'timestamp',
                                  'previous_bought_merchant_id',
                                  'label'])
input_df = pd.concat([chunk[chunk['label'] == 1] for chunk in iter_csv])
input_df.info()
len(input_df)
input_df.to_csv('./test_pd_iterator.csv', index=False)





wide_input_df = pd.read_csv(wide_input_file,
                            names=['user_id',
                                   'impress_merchant_id',
                                   'timestamp',
                                   'booked_merchant_id',
                                   'label'],
                            nrows=1000000)

wide_input_df.head()
tmp_df = wide_input_df.query('label>0')
tmp_df.head()
min_timestamp = min(wide_input_df['timestamp'])
max_timestamp = max(wide_input_df['timestamp'])
from datetime import datetime

min_datetime = datetime.fromtimestamp(min_timestamp/1000)
print(min_datetime)

max_datetime = datetime.fromtimestamp(max_timestamp/1000)
print(max_datetime)
wide_input_df['month'] = wide_input_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000).month)
wide_input_df['weekday'] = wide_input_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000).weekday())
wide_input_df['hour'] = wide_input_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000).hour)

wide_input_df.head(10)

unique_users = wide_input_df['user_id'].unique()
len(unique_users)

import os
os.environ['AIRFLOW_HOME'] = '/srv/airflow'
from airflow.hooks.postgres_hook import PostgresHook

user_metadata_query = """
SELECT
    customer_id AS user_id,
    gender,
    birth_date,
    first_topup_date,
    operator_name,
    phone_verified,
    email_verified,
    LOWER(REGEXP_REPLACE(SPLIT_PART(device, ' ', 1), '[^a-zA-Z]', '', 'g')) as device,
    os,
    home_area
FROM dm_itl.t_customer_data_foundation
WHERE customer_id =  ANY (%(unique_users)s);
"""
postgres_conn_id = 'BOB'
dbhook = PostgresHook(postgres_conn_id=postgres_conn_id)
conn = dbhook.get_conn()
params = {'unique_users': list(unique_users)}
user_metadata_df = pd.read_sql_query(user_metadata_query, con=conn, params=params)
user_metadata_df.head(20)

import numpy as np
unique_merchants = np.unique(np.concatenate((wide_input_df['impress_merchant_id'].unique(),
                                             wide_input_df['booked_merchant_id'].unique()), axis=0))
print(len(wide_input_df['impress_merchant_id'].unique()))
print(len(wide_input_df['booked_merchant_id'].unique()))
print(len(unique_merchants))

merchant_metadata_query = """
SELECT DISTINCT
    merchant_id,
    partner,
    service_area,
    service_zone,
	category
    collections,
    latitude,
    longitude
FROM dm_stg.t_gofood_restaurant
WHERE merchant_id =  ANY (%(unique_merchants)s)
LIMIT 20;
"""
params = {'unique_merchants': list(unique_merchants)}
merchant_metadata_df = pd.read_sql_query(merchant_metadata_query, con=conn, params=params)
merchant_metadata_df.head(20)
