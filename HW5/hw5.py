from sklearn.feature_extraction.text import TfidfVectorizer
# addd your import
import json
import pandas as pd
import numpy as np
from sklearn import metrics
from nltk.cluster import KMeansClusterer, cosine_distance, euclidean_distance
# Q2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial import distance

# Q1


def cluster_kmean(train_file, test_file):
    with open(train_file, 'r', encoding='utf-8') as f:
        jayson_train = json.load(f)

    with open(test_file, 'r', encoding='utf-8') as f:
        jayson_test = json.load(f)

    train = pd.DataFrame(jayson_train)
    # Initialize the TfidfVectorizer
    # Set min document frequency to 5
    tfidf_vect = TfidfVectorizer(stop_words="english", min_df=5)
    dtm = tfidf_vect.fit_transform(train[0])
    # set number of clusters
    num_clusters = 3
    clusterer_Cos = KMeansClusterer(num_clusters, distance=cosine_distance, repeats=20)
    clusterer_Euc = KMeansClusterer(num_clusters, distance=euclidean_distance, repeats=20)
    clusters_cos = clusterer_Cos.cluster(dtm.toarray(), assign_clusters=True)
    clusters_Euc = clusterer_Euc.cluster(dtm.toarray(), assign_clusters=True)

    test = pd.DataFrame(jayson_test)
    # Use the first label in the ground-truth label list of each test document
    t = []
    for i in test[1]:
        t.append(i[0])
    test['label'] = t
    # Make prediction on test sample
    test_dtm = tfidf_vect.transform(test[0])
    predicted_cos = [clusterer_Cos.classify(v) for v in test_dtm.toarray()]
    predicted_Euc = [clusterer_Euc.classify(v) for v in test_dtm.toarray()]
    # Create a dataframe with cluster id and ground truth label
    confusion_df_cos = pd.DataFrame(
        list(zip(test['label'].values, predicted_cos)), columns=["label", "cluster"])
    confusion_df_Euc = pd.DataFrame(
        list(zip(test['label'].values, predicted_Euc)), columns=["label", "cluster"])
    # Draw the crosstab table
    crosstab_cos = pd.crosstab(index=confusion_df_cos['cluster'], columns=confusion_df_cos['label'])
    crosstab_Euc = pd.crosstab(index=confusion_df_Euc['cluster'], columns=confusion_df_Euc['label'])
    # Draw the majority vote into dictionary
    majority_vote_cos = crosstab_cos.idxmax(axis=1, skipna=True).to_dict()
    majority_vote_Euc = crosstab_Euc.idxmax(axis=1, skipna=True).to_dict()
    # Map true label to cluster id
    predicted_target_cos = [majority_vote_cos[i] for i in predicted_cos]
    predicted_target_Euc = [majority_vote_Euc[i] for i in predicted_Euc]
    # Precision/recall/f-score for each label
    result_cos = metrics.classification_report(test["label"], predicted_target_cos)
    result_Euc = metrics.classification_report(test["label"], predicted_target_Euc)
    # Print out the result
    print('cosine')
    print(crosstab_cos)
    for i in majority_vote_cos:
        print('Cluster %d: Topic %s' % (i, majority_vote_cos[i]))
    print(result_cos)
    print('\nL2')
    print(crosstab_Euc)
    for i in majority_vote_Euc:
        print('Cluster %d: Topic %s' % (i, majority_vote_Euc[i]))
    print(result_Euc)
    return None

# Q2


def cluster_lda(train_file, test_file):

    topic_assign = None
    with open(train_file, 'r', encoding='utf-8') as f:
        jayson_train = json.load(f)

    with open(test_file, 'r', encoding='utf-8') as f:
        jayson_test = json.load(f)

    train_q2 = pd.DataFrame(jayson_train)
    test_q2 = pd.DataFrame(jayson_test)
    # Use the first label in the ground-truth label list of each test document
    t = []
    for i in test_q2[1]:
        t.append(i[0])
    test_q2['label'] = t
    tf_vectorizer = CountVectorizer(max_df=0.90, min_df=5, stop_words='english')
    tf = tf_vectorizer.fit_transform(train_q2[0])
    tf_feature_names = tf_vectorizer.get_feature_names()
    X_train = tf
    X_test = tf_vectorizer.transform(test_q2[0])

    # Run LDA
    # max_iter control the number of iterations
    # evaluate_every determines how often the perplexity is calculated
    # n_jobs is the number of parallel threads
    num_topics = 3
    lda = LatentDirichletAllocation(n_components=num_topics,
                                    max_iter=25, verbose=1,
                                    evaluate_every=1, n_jobs=1,
                                    random_state=0).fit(X_train)
    # Predict the topic distribution of each document in test_file and
    # select the topic with highest probability.
    topic_assign = lda.transform(X_test)
    test_topic = pd.DataFrame(topic_assign)
    # Make prediction on test sample
    predicted_lda = test_topic.idxmax(axis=1, skipna=True).values
    # Create a dataframe with cluster id and ground truth label
    confusion_df_lda = pd.DataFrame(
        list(zip(test_q2['label'].values, predicted_lda)), columns=["label", "cluster"])
    # Draw the crosstab table
    crosstab_lda = pd.crosstab(index=confusion_df_lda['cluster'], columns=confusion_df_lda['label'])
    # Draw the majority vote into dictionary
    majority_vote_lda = crosstab_lda.idxmax(axis=1, skipna=True).to_dict()
    # Map true label to cluster id
    predicted_target_lda = [majority_vote_lda[i] for i in predicted_lda]
    # Precision/recall/f-score for each label
    result_lda = metrics.classification_report(test_q2['label'], predicted_target_lda)

    # Print the result
    print(crosstab_lda)
    for i in majority_vote_lda:
        print('Cluster %d: Topic %s' % (i, majority_vote_lda[i]))
    print(result_lda)
    return topic_assign


def find_similar(doc_id, topic_assign):

    docs = None
    topic_mix = topic_assign
    df_test_topic = pd.DataFrame(topic_mix)
    # Use the topic_assign to compare the distence between target and others
    target = df_test_topic.iloc[doc_id, :]
    score = [1 - distance.cosine(target, df_test_topic.iloc[i, :])
             for i in range(len(df_test_topic))]
    df_test_topic['score'] = score
    dtm_sort = df_test_topic.sort_values(by=['score'], ascending=False)
    # Find the top three similar
    top_three_index = dtm_sort.index.values[1:4]
    docs = top_three_index
    return docs


if __name__ == "__main__":

    # Due to randomness, you won't get the exact result
    # as shown here, but your result should be close
    # if you tune the parameters carefully

    # Q1
    print("Q1")
    cluster_kmean('train_text.json', 'test_text.json')

    # Q2
    print("\nQ2")
    topic_assign = cluster_lda('train_text.json', 'test_text.json')
    doc_ids = find_similar(10, topic_assign)
    print("docs similar to {0}: {1}".format(10, doc_ids))
