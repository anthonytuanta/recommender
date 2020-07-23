import sys
import logging
import numpy as np
import pickle
import pandas as pd
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from sklearn.feature_extraction.text import CountVectorizer
from airflow import DAG
from airflow.models import Variable
from airflow.operators.slack_operator import SlackAPIPostOperator
from airflow.operators import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.hooks.postgres_hook import PostgresHook
from airflow.macros import datetime, timedelta
from techinasia.nlp.tokenizer import json_array_string_to_list

sys.path.append('/home/airflow/dags/work_experience_job_post_scores')

try:
    from utilities.text_processing import non_letter_removal
    from utilities.text_processing import text_to_words
    from utilities.feature_extraction import get_avgfeature_vec_tfidf
    from utilities.feature_extraction import create_bow_vectors
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

DAG_NAME = 'workexp_to_jobpost'
OUTPUT_DIR = '/home/airflow/tmp/work_experience_1/'

dag = DAG(DAG_NAME,
          default_args=default_args,
          max_active_runs=1,
          schedule_interval=None)


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

    job_post_titles = job_post_df['title']
    num_job_post_titles = len(job_post_titles)

    work_experience_titles = work_experience_df['job_title']
    num_work_experience_titles = len(work_experience_titles)

    clean_job_post_titles = []
    logging.info('Cleaning and parsing the job titles...\n')
    count = 0
    for title in job_post_titles:
        # If the index is evenly divisible by 1000, print a message
        if ((count + 1) % 1000 == 0):
            logging.info('job title %d of %d\n' % (count + 1,
                                                   num_job_post_titles))
        (words, tagged_words) = (text_to_words(title,
                                               remove_stopwords=False,
                                               use_lem=False))
        clean_job_post_titles.append(words)
        count += 1

    (vectorizer, job_post_title_features) = create_bow_vectors(clean_job_post_titles)

    files = []
    # store title bow
    with open(OUTPUT_DIR + 'title_bow.pkl', 'wb') as handle:
        pickle.dump(vectorizer, handle)

    clean_work_experience_titles = []
    logging.info('Cleaning and parsing the job titles...\n')
    count = 0
    for title in work_experience_titles:
        # If the index is evenly divisible by 1000, print a message
        if ((count + 1) % 1000 == 0):
            logging.info('work experience title %d of %d\n' %
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
        pd.DataFrame.from_records(job_post_title_features, columns=col_name)
    job_post_title_feature_df['job_postings_id'] = job_post_df['id']

    output_filename = OUTPUT_DIR + 'bow_job_post_title_features.csv'
    job_post_title_feature_df.to_csv(output_filename,
                                     index=False,
                                     encoding='utf-8')

    work_experience_title_feature_df = \
        pd.DataFrame.from_records(work_experience_title_features.toarray(),
                                  columns=col_name)
    work_experience_title_feature_df['id'] = work_experience_df['id']

    output_filename = OUTPUT_DIR + '/bow_work_experience_title_features.csv'
    work_experience_title_feature_df.to_csv(output_filename,
                                            index=False,
                                            encoding='utf-8')
    files.append(OUTPUT_DIR + 'title_bow.pkl')
    return files


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

    job_skills_sql = """
    SELECT *
    FROM job_skills
    WHERE is_deleted = False
    ORDER BY "order"
    """
    job_skills_df = dw1.get_pandas_df(job_skills_sql)
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

    count_vectorizer = \
        CountVectorizer(lowercase=False,
                        analyzer=json_array_string_to_list,
                        vocabulary=job_skills_df['name'].tolist(),
                        max_features=500)

    files = []
    # store the content
    with open(OUTPUT_DIR + 'skills_bow.pkl', 'wb') as handle:
        pickle.dump(count_vectorizer, handle)

    job_post_skills_features = count_vectorizer.fit_transform(job_skills)

    col_name = []
    for i in xrange(0, job_post_skills_features.shape[1]):
        col_name.append('feature_%s' % i)
    # put back in db temporaily for other tasks to access
    job_post_skill_feature_df = \
        pd.DataFrame.from_records(job_post_skills_features.toarray(),
                                  columns=col_name)

    job_post_skill_feature_df['job_postings_id'] = join_df['id_x']

    output_filename = OUTPUT_DIR + 'bow_job_post_skill_features.csv'
    job_post_skill_feature_df.to_csv(output_filename,
                                     index=False,
                                     encoding='utf-8')

    # users_skills
    work_experience_skills_features = count_vectorizer.fit_transform(users_skills)

    # put back in db temporaily for other tasks to access
    work_experience_skill_feature_df = \
        pd.DataFrame.from_records(work_experience_skills_features.toarray(),
                                  columns=col_name)

    work_experience_skill_feature_df['id'] = work_experience_join_df['id']

    output_filename = OUTPUT_DIR + 'bow_work_experience_skill_features.csv'
    work_experience_skill_feature_df.to_csv(output_filename,
                                            index=False,
                                            encoding='utf-8')
    files.append(OUTPUT_DIR + 'skills_bow.pkl')
    return files


def compute_description_feature(*args, **kwargs):
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
    ORDER BY id
    """
    work_experience_df = dw1.get_pandas_df(work_experience_sql)

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

    # flatten words_after_lemma_no_stopwords
    processed_descriptions_df['words_after_lemma_no_stopwords'] = \
        processed_descriptions_df['lemmatized_words_with_no_stopwords']\
        .map(lambda x: " ".join([val for sublist in x for val in sublist]))

    processed_descriptions_df.loc[
        processed_descriptions_df['words_after_lemma_no_stopwords'].isnull(),
        'words_after_lemma_no_stopwords'] = ""

    # lowercase
    processed_descriptions_df['words_after_lemma_no_stopwords'] = \
        processed_descriptions_df['words_after_lemma_no_stopwords']\
        .map(lambda x: non_letter_removal(x.lower()))

    processed_descriptions_df['lemmatized_sentences'] = \
        processed_descriptions_df['lemmatized_sentences']\
        .map(lambda x: [non_letter_removal(sublist.lower()) for sublist in x])

    processed_descriptions_df['lemmatized_sentences'] = \
        processed_descriptions_df["lemmatized_sentences"]\
        .map(lambda y: list(filter(lambda x: x != ' ', y)))

    sentences_tmp = processed_descriptions_df['lemmatized_sentences']
    sentences = []

    for description in sentences_tmp:
        for sentence in description:
            sentences.append(sentence.split())

    files = []
    # --------------------train word2vec--------------------------------
    # configure logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 500    # Word vector dimensionality
    min_word_count = 20   # Minimum word count
    num_workers = 2       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    logging.info('Training model...')

    # train model with POS and lemmatization
    model = word2vec.Word2Vec(sentences,
                              workers=num_workers,
                              size=num_features,
                              min_count=min_word_count,
                              window=context,
                              sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # save the model
    model_name = OUTPUT_DIR + 'w2v_500features_20minwords_10context'
    model.save(model_name)

    files.append(OUTPUT_DIR + 'w2v_500features_20minwords_10context')
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
        processed_work_summaries_df[
            'lemmatized_words_with_no_stopwords'].map(
            lambda x: " ".join([val for sublist in x for val in sublist]))

    processed_work_summaries_df\
        .loc[processed_work_summaries_df[
             'words_after_lemma_no_stopwords'].isnull(),
             'words_after_lemma_no_stopwords'] = ''
    processed_work_summaries_df['words_after_lemma_no_stopwords'] = \
        processed_work_summaries_df[
            'words_after_lemma_no_stopwords']\
        .map(lambda x: non_letter_removal(x.lower()))

    # ----------Creating features from Vector Averaging using Tfidf weights----
    selected_processed_descriptions_df = \
        processed_descriptions_df[
            processed_descriptions_df[
                  'job_posting_id'].isin(job_post_df['id'])]

    selected_descriptions = \
        selected_processed_descriptions_df['words_after_lemma_no_stopwords']

    clean_descriptions = \
        processed_descriptions_df['words_after_lemma_no_stopwords']

    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    vectorizer.fit(clean_descriptions)

    # need to save when running full model
    # store the content
    with open(OUTPUT_DIR + 'description_tfidf.pkl', 'wb') as handle:
        pickle.dump(vectorizer, handle)

    files.append(OUTPUT_DIR + 'description_tfidf.pkl')

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
        pd.DataFrame.from_records(datavecsavg_tfidf_job_post,
                                  columns=col_name)

    datavecsavg_tfidf_job_post_df['job_postings_id'] = \
        selected_processed_descriptions_df['job_posting_id']

    output_filename = OUTPUT_DIR + 'datavecsavg_tfidf_job_post_features.csv'
    datavecsavg_tfidf_job_post_df.to_csv(output_filename,
                                         index=False,
                                         encoding='utf-8')

    selected_processed_work_summaries_df = \
        processed_work_summaries_df[
            processed_work_summaries_df[
                'work_experience_id'].isin(work_experience_df['id'])]

    selected_summaries = \
        selected_processed_work_summaries_df['words_after_lemma_no_stopwords']

    datavecsavg_tfidf_work_experience = \
        get_avgfeature_vec_tfidf(selected_summaries,
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
    return files


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
                    'datavecsavg_tfidf_work_experience_features.csv',
                    header=0)

    combine_job_post_features_df = pd.merge(datavecsavg_tfidf_job_post_df,
                                            job_post_title_features_df,
                                            how='outer',
                                            on='job_postings_id')

    combine_job_post_features_df = pd.merge(combine_job_post_features_df,
                                            job_post_skill_features_df,
                                            how='outer',
                                            on='job_postings_id')

    combine_job_post_features_df = pd.merge(combine_job_post_features_df,
                                            job_post_df,
                                            left_on='job_postings_id',
                                            right_on='id')

    selected_post_id = combine_job_post_features_df['job_postings_id']

    combine_job_post_features_df = \
        combine_job_post_features_df.drop(['job_postings_id',
                                           'id',
                                           'title'], axis=1)

    combine_job_post_features_df = combine_job_post_features_df.fillna(0)
    datavecsavg_tfidf_job_post_stack = combine_job_post_features_df.as_matrix()

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
                 work_experience_df, left_on='id',
                 right_on='id')

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

    # compute cosine similarity
    scores_matrix = \
        pairwise.cosine_similarity(datavecsavg_tfidf_work_experience_stack,
                                   datavecsavg_tfidf_job_post_stack)

    # create dataframefrom cosine similarity matrix
    logging.info('Matrix size: %s x %s' % (scores_matrix.shape[0],
                                           scores_matrix.shape[1]))

    partnum = kwargs['params']['partnum']
    partindex = kwargs['params']['partindex']

    selected_indices = np.where(np.round(scores_matrix, 2) > 0.01)

    del combine_work_experience_features_df
    del datavecsavg_tfidf_work_experience_stack
    del datavecsavg_tfidf_job_post_stack
    del combine_job_post_features_df
    del work_experience_df
    del job_post_df
    del job_post_title_features_df
    del job_post_skill_features_df
    del datavecsavg_tfidf_job_post_df
    del work_experience_title_features_df
    del work_experience_skill_features_df
    del datavecsavg_tfidf_work_experience_df

    end_points = [0]
    for i in np.arange(partnum):
        end_points.append((i + 1) * len(selected_indices[0]) / partnum)

    scores = []
    for i in np.arange(end_points[partindex], end_points[partindex + 1]):
        if i % 1000000 == 0:
            logging.info('Processed %s job posts' % (i))
        work_id = selected_indices[0][i]
        job_id = selected_indices[1][i]
        v = scores_matrix[work_id, job_id]
        scores.append((selected_work_experience_id[work_id],
                       selected_work_experience_user_id[work_id],
                       selected_post_id[job_id],
                       v))
    scores_df = pd.DataFrame.from_records(scores,
                                          columns=['work_experience_id',
                                                   'user_id',
                                                   'similar_job_posting_id',
                                                   'score'])

    scores_df['model_id'] = pd.Series(6, index=scores_df.index)

    output_filename = \
        OUTPUT_DIR + \
        'scores_work_experience_to_job_posts_part%d.csv' % (partindex + 1)
    scores_df.to_csv(output_filename, index=False, encoding='utf-8')


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

create_temp_scores_table_op = PostgresOperator(
    task_id='create_temp_scores_table',
    postgres_conn_id='db1_etl',
    sql='scripts/postgres/create_temp_scores_table.sql',
    dag=dag
)

delete_temp_scores_table_op = PostgresOperator(
    task_id='delete_temp_scores_table',
    postgres_conn_id='db1_etl',
    sql='scripts/postgres/delete_temp_scores_table.sql',
    dag=dag
)

remove_scores_op = PostgresOperator(
    task_id='remove_scores',
    postgres_conn_id='db1_etl',
    sql='scripts/postgres/remove_work_experience_job_post_scores.sql',
    dag=dag
)

update_scores_op = PostgresOperator(
    task_id='update_scores',
    postgres_conn_id='db1_etl',
    sql='scripts/postgres/update_work_experience_job_post_scores.sql',
    dag=dag
)

dummy_op = DummyOperator(task_id='compute_similarity_branching', dag=dag)

copy_scores_to_temp_table_op = BashOperator(
    task_id='copy_scores_to_temp_table',
    bash_command='scripts/bash/copy_scores_to_temp_table.sh',
    params={"partnum": 4},
    provide_context=True,
    dag=dag)

for option in np.arange(4):
    t = PythonOperator(
       task_id='compute_similarity_branch_%d' % option,
       python_callable=compute_similarity_score,
       params={'partnum': 4, 'partindex': option},
       provide_context=True,
       pool='high_memory_usage',
       dag=dag)
    t.set_upstream(dummy_op)
    t.set_downstream(create_temp_scores_table_op)

archive_trained_models_op = BashOperator(
    task_id='archive_trained_models',
    bash_command='scripts/bash/archive_trained_models.sh',
    dag=dag
)

notify_processing_completion_op = SlackAPIPostOperator(
    task_id='notify_processing_completion',
    token=Variable.get('slack_token'),
    channel='#engineering-commits',
    username='TIA Airflow',
    icon_url=Variable.get('tia_slack_icon_url'),
    text='*user_work_experience_job_posting_similarity_scores* has been refreshed on {{ts}}',
    dag=dag
)

create_temp_scores_table_op.set_downstream(copy_scores_to_temp_table_op)
copy_scores_to_temp_table_op.set_downstream(remove_scores_op)
copy_scores_to_temp_table_op.set_downstream(update_scores_op)
delete_temp_scores_table_op.set_upstream(remove_scores_op)
delete_temp_scores_table_op.set_upstream(update_scores_op)
delete_temp_scores_table_op.set_downstream(notify_processing_completion_op)

dummy_op.set_upstream(compute_title_feature_op)
dummy_op.set_upstream(compute_skill_feature_op)
dummy_op.set_upstream(compute_description_feature_op)
dummy_op.set_downstream(archive_trained_models_op)
