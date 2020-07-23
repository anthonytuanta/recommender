import sys
import pandas as pd
import logging
import numpy as np
import pickle
import boto3
import gzip
from gensim.models import word2vec
from sklearn.metrics import pairwise
from airflow import DAG
from airflow.models import Variable
from airflow.operators.slack_operator import SlackAPIPostOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.hooks.postgres_hook import PostgresHook
from airflow.hooks.mysql_hook import MySqlHook
from airflow.macros import datetime, timedelta
sys.path.append('/home/airflow/dags/work_experience_job_post_scores')

try:
    from utilities.text_processing import non_letter_removal
    from utilities.text_processing import text_to_words
    from utilities.feature_extraction import get_avgfeature_vec_tfidf
except Exception as error:
    logging.error(error)
    raise Exception(error)

default_args = {
    'owner': 'tia_airflow',
    'email': ['dev+airflow@techinasia.com'],
    'start_date': datetime(2017, 6, 6),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'priority_weight': 10
}

DAG_NAME = 'workexp_to_jobpost_incremental'
INPUT_DIR = '/home/airflow/tmp/work_experience_1/'
OUTPUT_DIR = '/home/airflow/tmp/work_experience_1/incremental/'

MODEL_PATH = "machine_learning/model_id=6/"

dag = DAG(DAG_NAME,
          default_args=default_args,
          max_active_runs=1,
          schedule_interval=None)


def download_file_from_S3(S3_file, local_file):
    s3 = boto3.resource('s3')
    compressed_local_file = local_file + '.gz'

    s3.Bucket('techinasia-data').download_file(S3_file, compressed_local_file)

    with gzip.open(compressed_local_file, 'rb') as file_handler:
        content = file_handler.read()
        with open(local_file, 'w') as open_handler:
            open_handler.write(content)


def check_job_postings_to_be_updated(*args, **kwargs):
    dw1 = PostgresHook(postgres_conn_id='dw1_etl')

    job_post_sql = """
    SELECT *
    FROM job_postings
    WHERE is_deleted = False
    AND(
        title IS NOT NULL
        OR description IS NOT NULL
        OR skills_group_id IS NOT NULL
    )
    ORDER BY id
    """
    job_post_df = dw1.get_pandas_df(job_post_sql)

    deleted_job_post_sql = """
    SELECT id
    FROM job_postings
    WHERE is_deleted = True
    """
    deleted_job_post_df = dw1.get_pandas_df(deleted_job_post_sql)

    db1 = PostgresHook(postgres_conn_id='db1_etl')

    similarity_scores_sql = """
    SELECT DISTINCT
        similar_job_posting_id AS id,
        MAX(updated_at) AS last_updated_timestamp
    FROM user_work_experience_job_posting_similarity_scores
    GROUP BY 1
    """

    similarity_scores_df = db1.get_pandas_df(similarity_scores_sql)

    latest_update = similarity_scores_df.max()['last_updated_timestamp']

    new_job_post_df = \
        job_post_df.query('id not in @similarity_scores_df.id\
                          and last_updated_timestamp > @latest_update')

    to_remove_job_post_df = \
        similarity_scores_df.query('id in @deleted_job_post_df.id')

    processed_job_post_df = pd.merge(job_post_df,
                                     similarity_scores_df,
                                     on='id')

    to_update_job_post_df = \
        processed_job_post_df[
              (processed_job_post_df['last_updated_timestamp_x']
               > processed_job_post_df['last_updated_timestamp_y'])]

    output_df = pd.DataFrame()
    output_df['job_postings_id'] = new_job_post_df['id']
    output_df.to_csv(OUTPUT_DIR + 'to_add_job_posts.csv',
                     index=False,
                     encoding='utf-8')

    # output list to update
    output_df = pd.DataFrame()
    output_df['job_postings_id'] = to_update_job_post_df['id']
    output_df.to_csv(OUTPUT_DIR + 'to_update_job_posts.csv',
                     index=False,
                     encoding='utf-8')

    # output list to remove
    to_remove_df = pd.DataFrame()
    to_remove_df['job_postings_id'] = to_remove_job_post_df['id']
    to_remove_df.to_csv(OUTPUT_DIR + 'to_remove_job_posts.csv',
                        index=False,
                        encoding='utf-8')


def check_work_experiences_to_be_updated(*args, **kwargs):
    dw1 = PostgresHook(postgres_conn_id='dw1_etl')

    work_experience_sql = """
    SELECT id,
        updated_timestamp
    FROM users_work_experiences
    WHERE is_deleted = False
    AND(
        job_title IS NOT NULL
        OR summary IS NOT NULL
    )
    ORDER BY id
    """

    work_experience_df = dw1.get_pandas_df(work_experience_sql)

    deleted_work_experience_sql = """
    SELECT *
    FROM users_work_experiences
    WHERE is_deleted = True
    """

    deleted_work_experience_df = dw1.get_pandas_df(deleted_work_experience_sql)

    db1 = PostgresHook(postgres_conn_id='db1_etl')

    similarity_scores_sql = """
    SELECT DISTINCT
        work_experience_id AS id,
        MAX(updated_at) AS last_updated_timestamp
    FROM user_work_experience_job_posting_similarity_scores
    GROUP BY 1
    """

    similarity_scores_df = db1.get_pandas_df(similarity_scores_sql)

    latest_update = similarity_scores_df.max()['last_updated_timestamp']

    new_work_experience_df = \
        work_experience_df.query('id not in @similarity_scores_df.id\
                          and updated_timestamp > @latest_update')

    to_remove_work_experience_df = \
        similarity_scores_df.query('id in @deleted_work_experience_df.id')

    processed_work_experience_df = pd.merge(work_experience_df,
                                            similarity_scores_df,
                                            on='id')

    to_update_work_experience_df = \
        processed_work_experience_df[
             (processed_work_experience_df['updated_timestamp']
              > processed_work_experience_df['last_updated_timestamp'])]

    to_update_work_experience_df.drop(['updated_timestamp',
                                       'last_updated_timestamp'],
                                      axis=1, inplace=True)

    work_experience_with_skill_updated_sql = """
    SELECT
        uwe.id
    FROM users u
    JOIN users_work_experiences uwe ON uwe.user_id = u.id
    WHERE u.is_deleted = False
    AND u.skills_group_id IS NOT NULL
    AND u.skills_group_last_updated_timestamp > %(latest_update)s
    AND uwe.is_deleted = FALSE
    AND(
        uwe.job_title IS NOT NULL
        OR uwe.summary IS NOT NULL
    )
    ORDER BY uwe.id
    """

    params = {'latest_update': latest_update}

    work_experience_with_skill_updated_df = \
        dw1.get_pandas_df(work_experience_with_skill_updated_sql,
                          parameters=params)

    work_experience_with_skill_updated_df = \
        work_experience_with_skill_updated_df.\
        query('id not in @new_work_experience_df.id')

    to_update_work_experience_df = \
        pd.concat([to_update_work_experience_df,
                   work_experience_with_skill_updated_df]).drop_duplicates()

    output_df = pd.DataFrame()
    output_df['work_experience_id'] = new_work_experience_df['id']
    output_df.to_csv(OUTPUT_DIR + 'to_add_work_experiences.csv',
                     index=False,
                     encoding='utf-8')

    # output list to update
    output_df = pd.DataFrame()
    output_df['work_experience_id'] = to_update_work_experience_df['id']
    output_df.to_csv(OUTPUT_DIR + 'to_update_work_experiences.csv',
                     index=False,
                     encoding="utf-8")

    # output list to remove
    to_remove_df = pd.DataFrame()
    to_remove_df['work_experience_id'] = to_remove_work_experience_df['id']
    to_remove_df.to_csv(OUTPUT_DIR + 'to_remove_work_experiences.csv',
                        index=False,
                        encoding='utf-8')


def check_to_remove(*args, **kwargs):
    import os.path
    do_removing = False
    file_name = OUTPUT_DIR + 'to_remove_job_posts.csv'
    if os.path.exists(file_name):
        to_remove_job_post_df = pd.read_csv(file_name, header=0)
        if len(to_remove_job_post_df) > 0:
            do_removing = True

    file_name = OUTPUT_DIR + 'to_remove_work_experiences.csv'
    if os.path.exists(file_name):
        to_remove_job_post_df = pd.read_csv(file_name, header=0)
        if len(to_remove_job_post_df) > 0:
            do_removing = True

    if (do_removing):
        return 'remove_scores'
    else:
        return 'nothing_to_remove'


def check_to_update(*args, **kwargs):
    import os.path
    do_updating = False
    file_name = OUTPUT_DIR + 'to_add_job_posts.csv'
    if os.path.exists(file_name):
        to_remove_job_post_df = pd.read_csv(file_name, header=0)
        if len(to_remove_job_post_df) > 0:
            do_updating = True

    file_name = OUTPUT_DIR + 'to_update_job_posts.csv'
    if os.path.exists(file_name):
        to_remove_job_post_df = pd.read_csv(file_name, header=0)
        if len(to_remove_job_post_df) > 0:
            do_updating = True

    file_name = OUTPUT_DIR + 'to_add_work_experiences.csv'
    if os.path.exists(file_name):
        to_remove_job_post_df = pd.read_csv(file_name, header=0)
        if len(to_remove_job_post_df) > 0:
            do_updating = True

    file_name = OUTPUT_DIR + 'to_update_work_experiences.csv'
    if os.path.exists(file_name):
        to_remove_job_post_df = pd.read_csv(file_name, header=0)
        if len(to_remove_job_post_df) > 0:
            do_updating = True

    if (do_updating):
        return 'update_scores_branch'
    else:
        return 'nothing_to_update'


def compute_title_feature(*args, **kwargs):
    dw1 = PostgresHook(postgres_conn_id='dw1_etl')

    work_experience_sql = """
    SELECT *
    FROM users_work_experiences
    WHERE is_deleted = False
    AND job_title IS NOT NULL
    ORDER BY id
    """

    job_post_sql = """
    SELECT *
    FROM job_postings
    WHERE is_deleted = False
    AND title IS NOT NULL
    ORDER BY id
    """

    job_post_df = dw1.get_pandas_df(job_post_sql)

    work_experience_df = dw1.get_pandas_df(work_experience_sql)

    # get the titles
    job_post_titles = job_post_df['title']
    num_job_post_titles = len(job_post_titles)

    work_experience_titles = work_experience_df['job_title']
    num_work_experience_titles = len(work_experience_titles)

    to_update_job_df = pd.read_csv(OUTPUT_DIR + 'to_update_job_posts.csv',
                                   header=0)

    to_add_job_df = pd.read_csv(OUTPUT_DIR + 'to_add_job_posts.csv',
                                header=0)

    to_update_job_df = pd.merge(to_update_job_df,
                                to_add_job_df,
                                on='job_postings_id',
                                how='outer')

    to_update_work_experience_df = \
        pd.read_csv(OUTPUT_DIR + 'to_update_work_experiences.csv', header=0)

    to_add_work_experience_df = \
        pd.read_csv(OUTPUT_DIR + 'to_add_work_experiences.csv', header=0)

    to_update_work_experience_df = pd.merge(to_update_work_experience_df,
                                            to_add_work_experience_df,
                                            on='work_experience_id',
                                            how='outer')

    clean_job_post_titles = []
    logging.info('Cleaning and parsing the job titles...\n')
    count = 0
    for title in job_post_titles:
        if ((count + 1) % 1000 == 0):
            logging.info('description %d of %d\n' % (count + 1,
                                                     num_job_post_titles))
        (words, tagged_words) = (text_to_words(title,
                                               remove_stopwords=False,
                                               use_lem=False))
        clean_job_post_titles.append(words)
        count += 1

    S3_file = MODEL_PATH + 'title_bow.pkl.gz'
    local_file = INPUT_DIR + 'title_bow.pkl'
    download_file_from_S3(S3_file, local_file)

    vectorizer = pickle.load(open(local_file, 'rb'))

    job_post_title_features = vectorizer.transform(clean_job_post_titles)

    clean_work_experience_titles = []
    logging.info('Cleaning and parsing the job titles...\n')
    count = 0
    for title in work_experience_titles:
        if ((count + 1) % 1000 == 0):
            logging.info('description %d of %d\n' %
                         (count + 1, num_work_experience_titles))
        (words, tagged_words) = (text_to_words(title,
                                               remove_stopwords=False,
                                               use_lem=False))
        clean_work_experience_titles.append(words)
        count += 1

    # get work exp title feature
    work_experience_title_features = \
        vectorizer.transform(clean_work_experience_titles)

    col_name = []
    for i in xrange(0, job_post_title_features.shape[1]):
        col_name.append('feature_%s' % i)

    job_post_title_feature_df = \
        pd.DataFrame.from_records(job_post_title_features.toarray(),
                                  columns=col_name)
    job_post_title_feature_df['job_postings_id'] = job_post_df['id']

    output_filename = OUTPUT_DIR + 'bow_job_post_title_features.csv'
    job_post_title_feature_df.to_csv(output_filename,
                                     index=False,
                                     encoding='utf-8')

    work_experience_title_feature_df = \
        pd.DataFrame.from_records(work_experience_title_features.toarray(),
                                  columns=col_name)
    work_experience_title_feature_df['id'] = work_experience_df['id']

    output_filename = OUTPUT_DIR + 'bow_work_experience_title_features.csv'
    work_experience_title_feature_df.to_csv(output_filename,
                                            index=False,
                                            encoding='utf-8')


def compute_skill_feature(*args, **kwargs):
    dw1 = PostgresHook(postgres_conn_id='dw1_etl')
    job_post_sql = """
    SELECT *
    FROM job_postings
    WHERE is_deleted = False
    AND skills_group_id IS NOT NULL
    ORDER BY id
    """
    job_post_df = dw1.get_pandas_df(job_post_sql)

    skills_groups_sql = """SELECT * FROM skills_groups"""
    skills_groups_df = dw1.get_pandas_df(skills_groups_sql)

    join_df = pd.merge(job_post_df,
                       skills_groups_df,
                       left_on='skills_group_id',
                       right_on='id',
                       how='left')

    job_skills = join_df['list']

    work_experience_sql = """
    SELECT *
    FROM users_work_experiences
    WHERE is_deleted = False
    ORDER BY id
    """
    work_experience_df = dw1.get_pandas_df(work_experience_sql)

    users_sql = """
    SELECT *
    FROM users
    WHERE is_deleted = False
    AND skills_group_id IS NOT NULL
    ORDER BY id
    """
    users_df = dw1.get_pandas_df(users_sql)
    users_join_df = pd.merge(users_df,
                             skills_groups_df,
                             left_on='skills_group_id',
                             right_on='id',
                             how='left')

    work_experience_join_df = pd.merge(work_experience_df,
                                       users_join_df,
                                       left_on='user_id',
                                       right_on='id_x')
    users_skills = work_experience_join_df['list']

    S3_file = MODEL_PATH + 'skills_bow.pkl.gz'
    local_file = INPUT_DIR + 'skills_bow.pkl'
    download_file_from_S3(S3_file, local_file)

    count_vectorizer = pickle.load(open(local_file, 'rb'))

    job_post_skills_features = count_vectorizer.fit_transform(job_skills)

    work_experience_skills_features = \
        count_vectorizer.fit_transform(users_skills)

    col_name = []
    for i in xrange(0, job_post_skills_features.shape[1]):
        col_name.append('feature_%s' % i)

    job_post_skill_feature_df = \
        pd.DataFrame.from_records(job_post_skills_features.toarray(),
                                  columns=col_name)

    job_post_skill_feature_df['job_postings_id'] = join_df['id_x']

    output_filename = OUTPUT_DIR + 'bow_job_post_skill_features.csv'
    job_post_skill_feature_df.to_csv(output_filename,
                                     index=False,
                                     encoding='utf-8')

    work_experience_skill_feature_df = \
        pd.DataFrame.from_records(work_experience_skills_features.toarray(),
                                  columns=col_name)

    work_experience_skill_feature_df['id'] = work_experience_join_df['id']

    output_filename = OUTPUT_DIR + 'bow_work_experience_skill_features.csv'
    work_experience_skill_feature_df.to_csv(output_filename,
                                            index=False,
                                            encoding='utf-8')


def compute_description_feature(*args, **kwargs):
    # configure logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    dw1 = PostgresHook(postgres_conn_id='dw1_etl')

    job_post_sql = """
    SELECT *
    FROM job_postings
    WHERE is_deleted = False
    AND description IS NOT NULL
    ORDER BY id
    """
    job_post_df = dw1.get_pandas_df(job_post_sql)

    work_experience_sql = """
    SELECT *
    FROM users_work_experiences
    WHERE is_deleted = False
    AND summary IS NOT NULL
    ORDER BY id"""
    work_experience_df = dw1.get_pandas_df(work_experience_sql)

    # get processed sentences from db staging
    db1 = PostgresHook(postgres_conn_id='db1_etl')

    processed_descriptions_sql = """
    SELECT *
    FROM job_posting_description_meta
    ORDER BY job_posting_id
    """
    processed_descriptions_unpivot_df = \
        db1.get_pandas_df(processed_descriptions_sql)

    processed_descriptions_unpivot_df = \
        processed_descriptions_unpivot_df.drop(['created_at',
                                                'updated_at',
                                                'id',
                                                'content_type_metadata'],
                                               axis=1)
    # pivot back
    processed_descriptions_df = \
        processed_descriptions_unpivot_df.pivot(index='job_posting_id',
                                                columns='content_type',
                                                values='content').reset_index()

    processed_descriptions_df.sort_values(by='job_posting_id', inplace=True)

    processed_descriptions_df['words_after_lemma_no_stopwords'] = \
        processed_descriptions_df['lemmatized_words_with_no_stopwords']\
        .map(lambda x: " ".join([val for sublist in x for val in sublist]))

    processed_descriptions_df.loc[
        processed_descriptions_df['words_after_lemma_no_stopwords'].isnull(),
        'words_after_lemma_no_stopwords'] = ''

    # lowercase
    processed_descriptions_df['words_after_lemma_no_stopwords'] = \
        processed_descriptions_df['words_after_lemma_no_stopwords']\
        .map(lambda x: non_letter_removal(x.lower()))

    processed_descriptions_df['lemmatized_sentences'] = \
        processed_descriptions_df['lemmatized_sentences']\
        .map(lambda x: [non_letter_removal(sublist.lower()) for sublist in x])

    processed_descriptions_df['lemmatized_sentences'] = \
        processed_descriptions_df['lemmatized_sentences']\
        .map(lambda y: list(filter(lambda x: x != ' ', y)))

    # Set values for various parameters
    num_features = 500

    # load the model
    S3_file = MODEL_PATH + 'w2v_500features_20minwords_10context.gz'
    local_file = INPUT_DIR + 'w2v_500features_20minwords_10context'
    download_file_from_S3(S3_file, local_file)

    model = word2vec.Word2Vec.load(local_file)

    db1 = PostgresHook(postgres_conn_id='db1_etl')

    processed_work_summaries_sql = """
    SELECT *
    FROM user_work_experience_meta
    ORDER BY work_experience_id
    """
    processed_work_summaries_unpivot_df = \
        db1.get_pandas_df(processed_work_summaries_sql)

    processed_work_summaries_unpivot_df = \
        processed_work_summaries_unpivot_df.drop(['created_at',
                                                  'updated_at',
                                                  'id',
                                                  'content_type_metadata'],
                                                 axis=1)
    # pivot back
    processed_work_summaries_df = \
        processed_work_summaries_unpivot_df.pivot(index='work_experience_id',
                                                  columns='content_type',
                                                  values='content')\
        .reset_index()

    processed_work_summaries_df.sort_values(by='work_experience_id',
                                            inplace=True)

    processed_work_summaries_df['words_after_lemma_no_stopwords'] = \
        processed_work_summaries_df['lemmatized_words_with_no_stopwords']\
        .map(lambda x: " ".join([val for sublist in x for val in sublist]))

    processed_work_summaries_df.loc[
        processed_work_summaries_df['words_after_lemma_no_stopwords'].isnull(),
        'words_after_lemma_no_stopwords'] = ''

    # lowercase
    processed_work_summaries_df['words_after_lemma_no_stopwords'] = \
        processed_work_summaries_df['words_after_lemma_no_stopwords']\
        .map(lambda x: non_letter_removal(x.lower()))

    # --Creating features from Vector Averaging using Tfidf weights----
    selected_processed_descriptions_df = \
        processed_descriptions_df[
            processed_descriptions_df[
                'job_posting_id'].isin(job_post_df['id'])]

    selected_descriptions = \
        selected_processed_descriptions_df['words_after_lemma_no_stopwords']

    # load the content
    S3_file = MODEL_PATH + 'description_tfidf.pkl.gz'
    local_file = INPUT_DIR + 'description_tfidf.pkl'
    download_file_from_S3(S3_file, local_file)
    vectorizer = pickle.load(open(local_file, 'rb'))

    logging.info('Compute feature vectors\n')
    datavecsavg_tfidf_job_post = \
        get_avgfeature_vec_tfidf(selected_descriptions,
                                 model,
                                 vectorizer,
                                 num_features)

    col_name = []
    for i in xrange(0, datavecsavg_tfidf_job_post.shape[1]):
        col_name.append('feature_%s' % i)

    datavecsavg_tfidf_job_post_df = \
        pd.DataFrame.from_records(datavecsavg_tfidf_job_post, columns=col_name)

    datavecsavg_tfidf_job_post_df['job_postings_id'] = \
        selected_processed_descriptions_df['job_posting_id']

    output_filename = OUTPUT_DIR + 'datavecsavg_tfidf_job_post_features.csv'
    datavecsavg_tfidf_job_post_df.to_csv(output_filename,
                                         index=False,
                                         encoding='utf-8')

    # user work summary
    selected_processed_work_summaries_df = \
        processed_work_summaries_df[
            processed_work_summaries_df[
                'work_experience_id'].isin(work_experience_df['id'])]

    selected_work_summaries = \
        selected_processed_work_summaries_df['words_after_lemma_no_stopwords']

    datavecsavg_tfidf_work_experience = \
        get_avgfeature_vec_tfidf(selected_work_summaries,
                                 model,
                                 vectorizer,
                                 num_features)

    datavecsavg_tfidf_work_experience_df = \
        pd.DataFrame.from_records(datavecsavg_tfidf_work_experience,
                                  columns=col_name)

    datavecsavg_tfidf_work_experience_df['id'] = \
        selected_processed_work_summaries_df['work_experience_id'].values

    output_filename = \
        OUTPUT_DIR + 'datavecsavg_tfidf_work_experience_features.csv'
    datavecsavg_tfidf_work_experience_df.to_csv(output_filename,
                                                index=False,
                                                encoding='utf-8')


def compute_similarity_score(*args, **kwargs):
    dw1 = PostgresHook(postgres_conn_id='dw1_etl')

    job_post_sql = """
    SELECT
        id, title
    FROM job_postings
    WHERE is_deleted = False
    AND(
        description IS NOT NULL
        OR title IS NOT NULL
    )
    ORDER BY id
    """
    job_post_df = dw1.get_pandas_df(job_post_sql)

    work_experience_sql = """
    SELECT
        id, user_id, job_title, summary
    FROM users_work_experiences
    WHERE is_deleted = False
    AND(
        job_title IS NOT NULL
        OR summary IS NOT NULL
    )
    ORDER BY id
    """
    work_experience_df = dw1.get_pandas_df(work_experience_sql)

    job_post_title_features_df = \
        pd.read_csv(OUTPUT_DIR + 'bow_job_post_title_features.csv', header=0)

    work_experience_title_features_df = \
        pd.read_csv(OUTPUT_DIR + 'bow_work_experience_title_features.csv',
                    header=0)

    job_post_skill_features_df = \
        pd.read_csv(OUTPUT_DIR + 'bow_job_post_skill_features.csv', header=0)

    work_experience_skill_features_df = \
        pd.read_csv(OUTPUT_DIR + 'bow_work_experience_skill_features.csv',
                    header=0)

    datavecsavg_tfidf_job_post_df = \
        pd.read_csv(OUTPUT_DIR + 'datavecsavg_tfidf_job_post_features.csv',
                    header=0)

    datavecsavg_tfidf_work_experience_df = \
        pd.read_csv(OUTPUT_DIR +
                    'datavecsavg_tfidf_work_experience_features.csv', header=0)

    combine_job_post_features_df = \
        pd.merge(datavecsavg_tfidf_job_post_df,
                 job_post_title_features_df,
                 how='outer',
                 on='job_postings_id')
    combine_job_post_features_df = \
        pd.merge(combine_job_post_features_df,
                 job_post_skill_features_df,
                 how='outer',
                 on='job_postings_id')
    combine_job_post_features_df = \
        pd.merge(combine_job_post_features_df,
                 job_post_df,
                 left_on='job_postings_id',
                 right_on='id')

    # extract the new job post features first before dropping unused columns
    to_update_job_df = pd.read_csv(OUTPUT_DIR + 'to_update_job_posts.csv',
                                   header=0)
    to_add_job_df = pd.read_csv(OUTPUT_DIR + 'to_add_job_posts.csv', header=0)
    to_update_job_df = pd.merge(to_update_job_df,
                                to_add_job_df,
                                on='job_postings_id',
                                how='outer')
    combine_new_job_post_features_df = \
        combine_job_post_features_df[
            combine_job_post_features_df[
                'job_postings_id'].isin(to_update_job_df['job_postings_id'])]

    selected_new_post_id = combine_new_job_post_features_df['job_postings_id']
    combine_new_job_post_features_df = \
        combine_new_job_post_features_df.drop(['job_postings_id',
                                               'id',
                                               'title'], axis=1)

    combine_new_job_post_features_df = \
        combine_new_job_post_features_df.fillna(0)
    datavecsavg_tfidf_new_job_post_stack = \
        combine_new_job_post_features_df.as_matrix()

    combine_old_job_post_features_df = \
        combine_job_post_features_df[
            ~combine_job_post_features_df[
               'job_postings_id'].isin(to_update_job_df['job_postings_id'])]

    selected_old_post_id = combine_old_job_post_features_df['job_postings_id']
    combine_old_job_post_features_df = \
        combine_old_job_post_features_df.drop(['job_postings_id',
                                               'id',
                                               'title'], axis=1)

    combine_old_job_post_features_df = \
        combine_old_job_post_features_df.fillna(0)
    datavecsavg_tfidf_old_job_post_stack = \
        combine_old_job_post_features_df.as_matrix()

    del combine_job_post_features_df

    combine_work_experience_features_df = \
        pd.merge(datavecsavg_tfidf_work_experience_df,
                 work_experience_title_features_df,
                 how='outer',
                 on='id')
    combine_work_experience_features_df = \
        pd.merge(combine_work_experience_features_df,
                 work_experience_skill_features_df,
                 how='outer',
                 on='id')

    combine_work_experience_features_df = \
        pd.merge(combine_work_experience_features_df,
                 work_experience_df,
                 left_on='id',
                 right_on='id')

    to_update_work_experience_df = \
        pd.read_csv(OUTPUT_DIR + 'to_update_work_experiences.csv', header=0)

    to_add_work_experience_df = \
        pd.read_csv(OUTPUT_DIR + 'to_add_work_experiences.csv', header=0)

    to_update_work_experience_df = pd.merge(to_update_work_experience_df,
                                            to_add_work_experience_df,
                                            on='work_experience_id',
                                            how='outer')

    combine_new_work_experience_features_df = \
        combine_work_experience_features_df[
            combine_work_experience_features_df['id']
            .isin(to_update_work_experience_df['work_experience_id'])]

    selected_work_experience_id = combine_work_experience_features_df['id']
    selected_work_experience_user_id = \
        combine_work_experience_features_df['user_id']

    combine_work_experience_features_df = \
        combine_work_experience_features_df.drop(['id',
                                                  'user_id',
                                                  'job_title',
                                                  'summary'], axis=1)

    combine_work_experience_features_df = \
        combine_work_experience_features_df.fillna(0)
    datavecsavg_tfidf_work_experience_stack = \
        combine_work_experience_features_df.as_matrix()

    selected_new_work_experience_id = \
        combine_new_work_experience_features_df['id']
    selected_new_work_experience_user_id = \
        combine_new_work_experience_features_df['user_id']

    combine_new_work_experience_features_df = \
        combine_new_work_experience_features_df.drop(['id',
                                                      'user_id',
                                                      'job_title',
                                                      'summary'], axis=1)

    combine_new_work_experience_features_df = \
        combine_new_work_experience_features_df.fillna(0)
    datavecsavg_tfidf_new_work_experience_stack = \
        combine_new_work_experience_features_df.as_matrix()

    # compute cosine similarity
    scores_all_work_exp_new_job = []
    if datavecsavg_tfidf_new_job_post_stack.shape[0] > 0:
        scores_matrix_all_work_exp_new_job = \
            pairwise.cosine_similarity(datavecsavg_tfidf_work_experience_stack,
                                       datavecsavg_tfidf_new_job_post_stack)

        selected_indices = \
            np.where(np.round(scores_matrix_all_work_exp_new_job, 2) > 0.01)

        for i in np.arange(len(selected_indices[0])):
            if i % 1000000 == 0:
                logging.info('Processed %s work experiences vs job post' % (i))
            work_id = selected_indices[0][i]
            job_id = selected_indices[1][i]
            v = scores_matrix_all_work_exp_new_job[work_id, job_id]
            scores_all_work_exp_new_job\
                .append((selected_work_experience_id[work_id],
                         selected_work_experience_user_id[work_id],
                         selected_new_post_id.iloc[job_id],
                         v))

    scores_all_work_exp_new_job_df = \
        pd.DataFrame.from_records(scores_all_work_exp_new_job,
                                  columns=['work_experience_id',
                                           'user_id',
                                           'similar_job_postings_id',
                                           'score'])

    scores_new_work_exp_old_job = []
    if datavecsavg_tfidf_new_work_experience_stack.shape[0] > 0:
        scores_matrix_new_work_exp_old_job = \
            pairwise.cosine_similarity(datavecsavg_tfidf_new_work_experience_stack,
                                       datavecsavg_tfidf_old_job_post_stack)

        selected_indices = \
            np.where(np.round(scores_matrix_new_work_exp_old_job, 2) > 0.01)

        for i in np.arange(len(selected_indices[0])):
            if i % 1000000 == 0:
                logging.info('Processed %s work experiences vs job posts' % (i))
            work_id = selected_indices[0][i]
            job_id = selected_indices[1][i]
            v = scores_matrix_new_work_exp_old_job[work_id, job_id]

            scores_new_work_exp_old_job\
                .append((selected_new_work_experience_id.iloc[work_id],
                         selected_new_work_experience_user_id.iloc[work_id],
                         selected_old_post_id.iloc[job_id],
                         v))

    scores_new_work_exp_old_job_df = \
        pd.DataFrame.from_records(scores_new_work_exp_old_job,
                                  columns=['work_experience_id',
                                           'user_id',
                                           'similar_job_postings_id',
                                           'score'])

    scores_df = pd.concat([scores_all_work_exp_new_job_df,
                           scores_new_work_exp_old_job_df])
    scores_df['model_id'] = pd.Series(6, index=scores_df.index)

    output_filename = OUTPUT_DIR + 'scores_work_experience_to_job_posts.csv'
    scores_df.to_csv(output_filename, index=False, encoding='utf-8')


check_job_posting_to_be_updated_op = PythonOperator(
    task_id='check_job_postings_to_be_updated',
    python_callable=check_job_postings_to_be_updated,
    provide_context=True,
    dag=dag
)

check_work_experience_to_be_updated_op = PythonOperator(
    task_id='check_work_experiences_to_be_updated',
    python_callable=check_work_experiences_to_be_updated,
    provide_context=True,
    dag=dag
)

compute_similarity_op = PythonOperator(
    task_id='compute_similarity',
    python_callable=compute_similarity_score,
    provide_context=True,
    dag=dag
)

compute_description_feature_op = PythonOperator(
    task_id='compute_description_feature',
    python_callable=compute_description_feature,
    provide_context=True,
    dag=dag
)

compute_title_feature_op = PythonOperator(
    task_id='compute_title_feature',
    python_callable=compute_title_feature,
    provide_context=True,
    dag=dag
)

compute_skill_feature_op = PythonOperator(
    task_id='compute_skill_feature',
    python_callable=compute_skill_feature,
    provide_context=True,
    dag=dag
)

remove_scores_op = BashOperator(
    task_id='remove_scores',
    bash_command='scripts/bash/remove_work_experience_job_post_scores_incr.sh',
    dag=dag
)

update_scores_op = BashOperator(
    task_id='update_scores',
    bash_command='scripts/bash/update_work_experience_job_post_scores_incr.sh',
    dag=dag
)

notify_processing_completion_op = SlackAPIPostOperator(
    task_id='notify_processing_completion',
    token=Variable.get('slack_token'),
    channel='#engineering-commits',
    username='TIA Airflow',
    icon_url=Variable.get('tia_slack_icon_url'),
    text='*user_work_experience_job_posting_similarity_scores* has been updated on {{ts}}',
    trigger_rule='all_done',
    dag=dag
)

check_to_remove_op = BranchPythonOperator(
    task_id='check_to_remove',
    python_callable=check_to_remove,
    provide_context=True,
    dag=dag
)

check_to_update_op = BranchPythonOperator(
    task_id='check_to_update',
    python_callable=check_to_update,
    provide_context=True,
    dag=dag
)

update_scores_branch_op = DummyOperator(
    task_id='update_scores_branch',
    dag=dag
)
nothing_to_remove_op = DummyOperator(
    task_id='nothing_to_remove',
    dag=dag
)

nothing_to_update_op = DummyOperator(
    task_id='nothing_to_update',
    dag=dag
)

check_job_posting_to_be_updated_op.set_downstream(check_to_remove_op)
check_job_posting_to_be_updated_op.set_downstream(check_to_update_op)

check_work_experience_to_be_updated_op.set_downstream(check_to_remove_op)
check_work_experience_to_be_updated_op.set_downstream(check_to_update_op)

update_scores_branch_op.set_upstream(check_to_update_op)
remove_scores_op.set_upstream(check_to_remove_op)
nothing_to_remove_op.set_upstream(check_to_remove_op)
nothing_to_update_op.set_upstream(check_to_update_op)

notify_processing_completion_op.set_upstream(nothing_to_remove_op)
notify_processing_completion_op.set_upstream(nothing_to_update_op)

update_scores_branch_op.set_downstream(compute_title_feature_op)
update_scores_branch_op.set_downstream(compute_skill_feature_op)
update_scores_branch_op.set_downstream(compute_description_feature_op)

compute_similarity_op.set_upstream(compute_title_feature_op)
compute_similarity_op.set_upstream(compute_skill_feature_op)
compute_similarity_op.set_upstream(compute_description_feature_op)
compute_similarity_op.set_downstream(update_scores_op)
notify_processing_completion_op.set_upstream(update_scores_op)
notify_processing_completion_op.set_upstream(remove_scores_op)
