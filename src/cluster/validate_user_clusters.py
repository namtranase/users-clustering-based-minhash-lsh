"""Validate user cluster program based on Minhasing LSH algorithm.
"""
import os
import glob
import logging
import multiprocessing
from functools import partial

import pandas as pd
import tqdm
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import settings
from src.config.config import read_config_file
from src.utils import memcache
from src.utils.minhash import MinHash, MinHashLSH


def load_data_file(fname):
    """Load data file and return dataframe.
    """
    print(f"Loading data from file [{fname}]")
    data_frame = pd.read_csv(fname)

    return data_frame

def load_user_activities_data(data_dir, workers=5):
    """Load data file and return dataframe.
    """
    pattern = os.path.join(data_dir, "*.csv")
    fnames = [fname for fname in glob.glob(pattern)]
    with multiprocessing.Pool(processes=workers) as pool:
        dfs = pool.map(load_data_file, fnames)

    return pd.concat(dfs)

def collect_users_posts(users_logs):
    """Filter and collect user logs.
    Update for event later.
    """
    # TODO: filter users have few logs
    users_posts = dict()
    for log in users_logs:
        user_id, post_id, _ = log
        user_id = str(user_id)
        post_id = str(post_id)
        if user_id not in users_posts:
            users_posts[user_id] = set()
        users_posts[user_id].update([post_id])

    return users_posts

def split_train_test(users_posts, fraction=0.8):
    """Split train test set.
    """
    train_set = dict()
    test_set = dict()
    for user, posts in users_posts.items():
        posts = list(posts)
        train_set[user] = posts[:int(len(posts)*fraction)]
        test_set[user] = posts[int(len(posts)*fraction):]

    return train_set, test_set

def encode_event_post(event_post):
    """Encode for various type of event posts.
    """
    # TODO: update for various type of events
    return event_post.encode('utf-8')

def simple_min_hashing(
        lsh, band, row, default_cache, users_posts):
    """Create minhash signature for user and build LSH Minhash.
    """
    # Create minhash for users, TODO: update using map function
    users_sigs = list()
    for user_id, posts in users_posts.items():
        sig = MinHash(num_perm=band*row, cache=default_cache)
        for post in posts:
            sig.update(encode_event_post(post))
        users_sigs.append([user_id, sig])

    # Update LSH Minhash based on insertion season
    with lsh.insertion_session() as session:
        for user_id, sig in users_sigs:
            session.insert(user_id, sig)

    return lsh

def build_cluster_map(hash_table):
    """Build cluster map based on buckets of hash table.
    """
    cluster_map = dict()
    for key in hash_table.keys():
        bucket = hash_table.get(key)
        # prune away clusters with low membership
        if len(bucket) > 3:
            cluster_map[key] = bucket

    return cluster_map

def buid_user_clusters(cluster_map):
    """Build user clusters from cluster map.
    """
    user_clusters = dict()
    for key, value in cluster_map.items():
        for user_id in value:
            if user_id not in user_clusters:
                user_clusters[user_id] = set()
            user_clusters[user_id].add(key)

    return user_clusters

def process_users_clusters(
        band, row, default_cache, users_posts, workers=10):
    """Process logs and build user clusters data.
    """

    # Build Minhash LSH and get hash tables
    lsh = MinHashLSH(num_perm=band*row,
                     params=(band, row))
    updated_lsh = simple_min_hashing(
        lsh, band, row, default_cache, users_posts)
    hash_tables = updated_lsh.hashtables
    logging.debug("Number of hash tables: %s", len(hash_tables))

    # Build bucket map from hash tables
    with multiprocessing.Pool(processes=workers) as pool:
        total_cluster_map = pool.map(build_cluster_map, hash_tables)
    cluster_map = total_cluster_map[0]
    for clusters in total_cluster_map:
        cluster_map.update(clusters)
    logging.debug("Number of clusters: %s", len(cluster_map))

    # Build user clusters from bucket map
    user_clusters = buid_user_clusters(cluster_map)
    logging.debug("Number of user have clusters: %s", len(user_clusters))

    return cluster_map, user_clusters

def proportional_user_user(num_cluster, cluster_ids, user_clusters):
    """Proportional user in the same clusters.
    """
    count_same_cluster = 0
    for cluster in cluster_ids:
        if cluster in user_clusters:
            count_same_cluster += 1
    return count_same_cluster / num_cluster

def cal_score_for_post(user_clusters, clusters,
                       user_same_cluster_posts, post):
    """Calulate score for post.
    """
    total_clusters = set()
    for key in user_same_cluster_posts.keys():
        total_clusters.update(user_clusters[key])
    score = list()
    for user, user_posts in user_same_cluster_posts.items():
        if post in user_posts:
            val_user_clusters = user_clusters[user]
            user_score = proportional_user_user(
                len(total_clusters), clusters, val_user_clusters)
            score.append(user_score)
        else: score.append(0)

    return sum(score)

def validate_test_set(users_posts, test_users_posts,
                      cluster_map, user_clusters):
    """Validate for test set.
    """
    true_l = list()
    predict_l = list()
    for user, posts in users_posts:
        # Check user clusters
        clusters = user_clusters.get(user, None)
        if clusters is None:
            # logging.debug("User don't have clusters")
            continue

        # Build post lables for user
        lables = [1] * len(posts)

        # Get users with same clusters and their posts
        user_same_cluster = set()
        for cluster_id in clusters:
            user_same_cluster.update(cluster_map[cluster_id])
        user_same_cluster_posts = dict()
        for user_id in user_same_cluster:
            user_posts = test_users_posts.get(user_id, None)
            if user_posts is not None:
                user_same_cluster_posts[user_id] = user_posts

        # Calculate score for each post
        post_scores = list()
        for post in posts:
            score_post = cal_score_for_post(
                user_clusters, clusters,
                user_same_cluster_posts, post)
            post_scores.append(score_post)

        # Save true lables and predict lables
        true_l.extend(lables)
        predict_l.extend(post_scores)

    return (true_l, predict_l)

def cal_metric_scores(
        true_lables, predict_lables, predict_threshold_lables):
    """Calculate score based on true and predict lables.
    Metrics: rmse, accuracy, recall, f1.
    """
    rmse = mean_squared_error(true_lables, predict_lables)
    accuracy = accuracy_score(true_lables, predict_threshold_lables)
    precison = precision_score(true_lables, predict_threshold_lables)
    recall = recall_score(true_lables, predict_threshold_lables)
    f1 = f1_score(true_lables, predict_threshold_lables)

    return dict(rmse_score=rmse,
                accuracy_score=accuracy,
                precision_score=precison,
                recall_score=recall,
                f1_score=f1)

def process_logs(config):
    """Process logs and build user clusters data.
    """
    # Read user activities csv files.
    workers = config['workers']
    data_dir = 'data/user_cluster'
    user_logs_df = load_user_activities_data(data_dir, workers=workers)

    # Create lists of users activities.
    total_users_logs = list()
    for _, rows in user_logs_df.iterrows():
        total_users_logs.append((rows.userId,
                                 rows.postId,
                                 rows.eventId))
    logging.info("Total of recent users activities: %s",
                 len(total_users_logs))

    # Train test split
    users_posts = collect_users_posts(total_users_logs)
    train_users_posts, test_users_posts = split_train_test(users_posts)
    logging.debug("Number of users after collect activities: %s",
                  len(train_users_posts))

    # Validate cluster Minhash LSH algorithm
    model_results = {'Num_band': [],
                     'Num_row': [],
                     'Threshold_post': [],
                     'Accuracy_score': [],
                     'Precision_score': [],
                     'Recall_score': [],
                     'F1_score': [],
                     'RMSE_score': []}

    # Grid Search, parametes: bands, rows and threshold for post score
    num_bands = range(10, 50, 5)
    num_rows = range(2, 10, 2)
    threshold_post = [10**-(exp/10) for exp in range(1, 10, 1)]

    pbar = tqdm.tqdm(
        total=len(num_bands)*len(num_rows)*len(threshold_post))
    # Iterate band range.
    for band in num_bands:
        # Iterate through row values.
        for row in num_rows:
            default_cache = memcache.LRUCache(capacity=500000)

            # Build user cluster based on MinHash algorithm
            cluster_map, user_clusters = process_users_clusters(
                band, row, default_cache, train_users_posts)

            # Calulate metric scrore for validate cluster
            # TODO: Xem lai cach tinh score cho user
            logging.debug("Calculate score for test set...")
            # Divide user_ids into chunks with chunk_size.
            users_posts = list()
            for user, posts in test_users_posts.items():
                users_posts.append([user, posts])
            chunk_size = 100
            chunks = [users_posts[i:i+chunk_size]
                        for i in range(0, len(users_posts), chunk_size)]
            with multiprocessing.Pool(processes=workers) as pool:
                results = pool.map(partial(validate_test_set,
                                            test_users_posts=test_users_posts,
                                            cluster_map=cluster_map,
                                            user_clusters=user_clusters),
                                    chunks)
            true_lables = list()
            predict_lables = list()
            for true_l, predict_l in results:
                true_lables.extend(true_l)
                predict_lables.extend(predict_l)

            # Iterate through row threshold value post.
            for threshold in threshold_post:
                logging.info("Building user clusters data with \
                            band: %s, row: %s and threshold: %s",
                            band, row, threshold)
                predict_threshold_lables = list()
                for score in predict_lables:
                    if score > threshold:
                        predict_threshold_lables.append(1)
                    else:
                        predict_threshold_lables.append(0)
                # Calculate metric scores
                result_scores = cal_metric_scores(
                    true_lables, predict_lables, predict_threshold_lables)

                logging.debug("Results are comming...")
                logging.info("Accuracy: %s, Precision: %s, Recall: %s, F1: %s, RMSE: %s",
                                result_scores['accuracy_score'],
                                result_scores['precision_score'],
                                result_scores['recall_score'],
                                result_scores['f1_score'],
                                result_scores['rmse_score'])

                # Update results
                model_results['Num_band'].append(band)
                model_results['Num_row'].append(row)
                model_results['Threshold_post'].append(threshold)
                model_results['Accuracy_score'].append(result_scores['accuracy_score'])
                model_results['Precision_score'].append(result_scores['precision_score'])
                model_results['Recall_score'].append(result_scores['recall_score'])
                model_results['F1_score'].append(result_scores['f1_score'])
                model_results['RMSE_score'].append(result_scores['rmse_score'])
                # model_results['Jaccard_score'].append(jaccard_score)
                pbar.update(1)

    pd.DataFrame(model_results).to_csv(
        "data/user_cluster/tuning_results.csv", index=False)
    pbar.close()

def main():
    """Main program
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)

    logging.info("Start user cluster with config: %s", config)
    process_logs(config)
    logging.info("Program was terminated!")

if __name__ == '__main__':
    main()
