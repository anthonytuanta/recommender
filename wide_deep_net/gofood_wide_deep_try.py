from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf
from datetime import datetime

#load and prepare data
input_file = './final_features_central_jakatar.csv'

# names=['customer_id',
#       'recommended_merchant_id',
#       'timestamp',
#       'previous_bought_merchant_id',
#       'label',
#       'operator_name',
#       'phone_verified',
#       'email_verified',
#       'os',
#       'os_version',
#       'app_version',
#       'device',
#       'home_area',
#       'very_first_service',
#       'very_first_service_gopay',
#       'gender',
#       'partner_recommended',
#       'service_area_recommended',
#       'service_zone_recommended',
#       'category_recommended',
#       'latitude_recommended',
#       'longitude_recommended',
#       'partner_bought',
#       'service_area_bought',
#       'service_zone_bought',
#       'category_bought',
#       'latitude_bought',
#       'longitude_bought'])

input_df = pd.read_csv(input_file, header=0)

# input_df['gender'] = input_df['partner_recommended'].apply(lambda x: x[0])
# input_df['partner_recommended'] = \
    # input_df['partner_recommended'].apply(lambda x: x[1])

input_df['datetime'] = \
    input_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000))
input_df['month'] = input_df['datetime'].apply(lambda x: x.month)
input_df['weekday'] = input_df['datetime'].apply(lambda x: x.weekday())
input_df['hour'] = input_df['datetime'].apply(lambda x: x.hour)
import re

#fill null as 'missing' for first services
input_df['very_first_service'] = \
    input_df['very_first_service'].fillna('missing')
input_df['very_first_service_gopay'] = \
    input_df['very_first_service_gopay'].fillna('missing')

#some users with missing gender, need to drop for now
input_df = input_df.dropna(how='any', axis=0)

input_df['device_processed'] = \
    input_df['device'].apply(lambda x: re.sub("[^a-z]",
                                              "",
                                              x.split()[0].lower()))

COLUMNS = ['customer_id', 'recommended_merchant_id',
           'previous_bought_merchant_id', 'operator_name', 'phone_verified',
           'email_verified', 'os', 'os_version', 'app_version',
           'device_processed', 'home_area', 'very_first_service',
           'very_first_service_gopay', 'gender', 'partner_recommended',
           'service_area_recommended', 'service_zone_recommended',
           'category_recommended', 'latitude_recommended',
           'longitude_recommended', 'partner_bought', 'service_area_bought',
           'service_zone_bought', 'category_bought', 'latitude_bought',
           'longitude_bought', 'month', 'weekday', 'hour']

LABEL_COLUMN = 'label'

CATEGORICAL_COLUMNS = ['recommended_merchant_id',
                       'previous_bought_merchant_id', 'operator_name',
                       'phone_verified', 'email_verified', 'os', 'os_version',
                       'app_version', 'device_processed', 'home_area',
                       'very_first_service', 'very_first_service_gopay',
                       'gender', 'partner_recommended',
                       'service_area_recommended', 'service_zone_recommended',
                       'category_recommended', 'partner_bought',
                       'service_area_bought', 'service_zone_bought',
                       'category_bought', 'month', 'weekday', 'hour']
CONTINUOUS_COLUMNS = ['latitude_recommended', 'longitude_recommended',
                      'latitude_bought', 'longitude_bought']

def build_estimator(model_dir, model_type):
    """Build an estimator."""
    # Sparse base columns.
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                       keys=["F", "M"])
    phone_verified = tf.contrib.layers.sparse_column_with_keys(column_name="phone_verified",
                                                       keys=["1", "0"])
    email_verified = tf.contrib.layers.sparse_column_with_keys(column_name="email_verified",
                                                       keys=["1", "0"])
    partner_recommended = tf.contrib.layers.sparse_column_with_keys(column_name="partner_recommended",
                                                       keys=["Y", "N"])
    partner_bought = tf.contrib.layers.sparse_column_with_keys(column_name="partner_bought",
                                                   keys=["Y", "N"])

    recommended_merchant_id = tf.contrib.layers.sparse_column_with_hash_bucket(
        "recommended_merchant_id", hash_bucket_size=1000)
    previous_bought_merchant_id = tf.contrib.layers.sparse_column_with_hash_bucket(
        "previous_bought_merchant_id", hash_bucket_size=1000)
    operator_name = tf.contrib.layers.sparse_column_with_hash_bucket(
        "operator_name", hash_bucket_size=100)
    os = tf.contrib.layers.sparse_column_with_hash_bucket(
      "os", hash_bucket_size=10)
    os_version = tf.contrib.layers.sparse_column_with_hash_bucket(
        "os_version", hash_bucket_size=100)
    app_version = tf.contrib.layers.sparse_column_with_hash_bucket(
        "app_version", hash_bucket_size=100)
    device = tf.contrib.layers.sparse_column_with_hash_bucket(
        "device_processed", hash_bucket_size=100)
    home_area = tf.contrib.layers.sparse_column_with_hash_bucket(
        "home_area", hash_bucket_size=100)
    very_first_service = tf.contrib.layers.sparse_column_with_hash_bucket(
        "very_first_service", hash_bucket_size=100)
    very_first_service_gopay = tf.contrib.layers.sparse_column_with_hash_bucket(
        "very_first_service_gopay", hash_bucket_size=100)
    service_area_recommended = tf.contrib.layers.sparse_column_with_hash_bucket(
        "service_area_recommended", hash_bucket_size=100)
    service_zone_recommended = tf.contrib.layers.sparse_column_with_hash_bucket(
        "service_zone_recommended", hash_bucket_size=1000)
    category_recommended = tf.contrib.layers.sparse_column_with_hash_bucket(
        "category_recommended", hash_bucket_size=1000)
    service_area_bought = tf.contrib.layers.sparse_column_with_hash_bucket(
        "service_area_bought", hash_bucket_size=100)
    service_zone_bought = tf.contrib.layers.sparse_column_with_hash_bucket(
        "service_zone_bought", hash_bucket_size=1000)
    category_bought = tf.contrib.layers.sparse_column_with_hash_bucket(
        "category_bought", hash_bucket_size=1000)
    month = tf.contrib.layers.sparse_column_with_hash_bucket(
        "month", hash_bucket_size=100)
    weekday = tf.contrib.layers.sparse_column_with_hash_bucket(
        "weekday", hash_bucket_size=100)
    hour = tf.contrib.layers.sparse_column_with_hash_bucket(
        "hour", hash_bucket_size=100)

      # Continuous base columns.
    latitude_recommended = tf.contrib.layers.real_valued_column("latitude_recommended")
    longitude_recommended = tf.contrib.layers.real_valued_column("longitude_recommended")
    latitude_bought = tf.contrib.layers.real_valued_column("latitude_bought")
    longitude_bought = tf.contrib.layers.real_valued_column("longitude_bought")

      # Transformations.
      # age_buckets = tf.contrib.layers.bucketized_column(age,
      #                                                   boundaries=[
      #                                                       18, 25, 30, 35, 40, 45,
      #                                                       50, 55, 60, 65
      #                                                   ])

    # Wide columns and deep columns.
    wide_columns = [
        gender, phone_verified, email_verified, partner_recommended,
        partner_bought, recommended_merchant_id, previous_bought_merchant_id,
        operator_name, os, os_version, app_version, device, home_area,
        very_first_service, very_first_service_gopay, service_area_recommended,
        service_zone_recommended, category_recommended, service_area_bought,
        service_zone_bought, category_bought, month, weekday, hour,
        tf.contrib.layers.crossed_column(
            [recommended_merchant_id, previous_bought_merchant_id],
            hash_bucket_size=int(1e4)),
        tf.contrib.layers.crossed_column(
            [category_recommended, category_bought],
            hash_bucket_size=int(1e4))
    ]
    deep_columns = [
        tf.contrib.layers.embedding_column(gender, dimension=32),
        tf.contrib.layers.embedding_column(phone_verified, dimension=32),
        tf.contrib.layers.embedding_column(email_verified, dimension=32),
        tf.contrib.layers.embedding_column(partner_recommended, dimension=32),
        tf.contrib.layers.embedding_column(partner_bought, dimension=32),
        tf.contrib.layers.embedding_column(recommended_merchant_id, dimension=32),
        tf.contrib.layers.embedding_column(previous_bought_merchant_id, dimension=32),
        tf.contrib.layers.embedding_column(operator_name, dimension=32),
        tf.contrib.layers.embedding_column(os, dimension=32),
        tf.contrib.layers.embedding_column(os_version, dimension=32),
        tf.contrib.layers.embedding_column(app_version, dimension=32),
        tf.contrib.layers.embedding_column(device, dimension=32),
        tf.contrib.layers.embedding_column(home_area, dimension=32),
        tf.contrib.layers.embedding_column(very_first_service, dimension=32),
        tf.contrib.layers.embedding_column(very_first_service_gopay, dimension=32),
        tf.contrib.layers.embedding_column(service_area_recommended, dimension=32),
        tf.contrib.layers.embedding_column(service_zone_recommended, dimension=32),
        tf.contrib.layers.embedding_column(category_recommended, dimension=32),
        tf.contrib.layers.embedding_column(service_area_bought, dimension=32),
        tf.contrib.layers.embedding_column(service_zone_bought, dimension=32),
        tf.contrib.layers.embedding_column(category_bought, dimension=32),
        tf.contrib.layers.embedding_column(month, dimension=32),
        tf.contrib.layers.embedding_column(weekday, dimension=32),
        tf.contrib.layers.embedding_column(hour, dimension=32),
        latitude_recommended,
        longitude_recommended,
        latitude_bought,
        longitude_bought
#       distance from recommended merchant to bought merchant
    ]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[500, 100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[500, 100, 50],
            fix_global_step_increment_bug=True)
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
    k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
    for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


# def train_and_eval(model_dir, model_type, train_steps, input_data):
    # """Train and evaluate the model."""
    # train_file_name, test_file_name = maybe_download(train_data, test_data)
df_train = input_df
df_test = input_df
# remove NaN elements
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)

for f in CATEGORICAL_COLUMNS:
    df_train[f] = df_train[f].astype(str)
    df_test[f] = df_test[f].astype(str)

df_train[LABEL_COLUMN] = df_train[LABEL_COLUMN].astype(int)
df_test[LABEL_COLUMN] = df_test[LABEL_COLUMN].astype(int)

model_dir = '/home/anthonyta/testing/wide_deep_net/model/'
model_type = 'wide_n_deep'
train_steps = 200
model_dir = tempfile.mkdtemp() if not model_dir else model_dir
print("model directory = %s" % model_dir)

m = build_estimator(model_dir, model_type)
m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
