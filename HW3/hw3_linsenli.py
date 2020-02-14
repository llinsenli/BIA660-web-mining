import nltk
from nltk.corpus import stopwords
import csv
from scipy.spatial import distance
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

# Q1 function


def tokenize(text):

    tokens = None
    text_lower = text.lower()
    stop_words = set(stopwords.words('english'))
    pattern = r'[a-zA-Z]+[-\._\']*[a-zA-Z]+'
    token1 = nltk.regexp_tokenize(text_lower, pattern)
    tokens = [i for i in token1 if i not in stop_words]
    return tokens

# Q2 function_1


def sentiment_analysis(text, positive_words, negative_words):

    negations = ["not", "no", "isn't", "wasn't", "aren't",
                 "weren't", "don't", "didn't", "cannot",
                 "couldn't", "won't", "neither", "nor"]
    sentiment = None
    tokens = tokenize(text)
    length = len(tokens)
    i = 1
    number_of_positive_words = 0
    number_of_negative_words = 0
    while i < length:
        if tokens[i] in positive_words:
            # A positive word not preceded by a negation word
            if tokens[i-1] not in negations:
                number_of_positive_words += 1
            # A positive word preceded by a negation word
            elif tokens[i-1] in negations:
                number_of_negative_words += 1

        elif tokens[i] in negative_words:
            # A negative word not preceded by a negation word
            if tokens[i-1] not in negations:
                number_of_negative_words += 1
            # A negative word preceded by a negation word
            elif okens[i-1] in negations:
                number_of_positive_words += 1
        i += 1
    if number_of_positive_words > number_of_negative_words:
        sentiment = 2
    else:
        sentiment = 1
    return sentiment

# Q2 function_2


def performance_evaluate(input_file, positive_words, negative_words):

    accuracy = None
    amazon_review = pd.read_csv(input_file)
    lis = [i for i in amazon_review.review]
    result = [sentiment_analysis(i, positive_words, negative_words) for i in lis]
    accuracy = np.sum(result == amazon_review.label)/len(lis)
    return accuracy

# Q3 function


def find_similar_doc(docs,  doc_id):
    top_sim_index, top_sim_score = None, None
    # Use the Q1 function to tokenize each doc in docs
    tokens = [tokenize(i) for i in docs]
    dic = [nltk.FreqDist(token) for token in tokens]
    docs_tokens = {idx: doc for idx, doc in enumerate(dic)}

    # Get document-term matrix
    dtm = pd.DataFrame.from_dict(docs_tokens, orient="index")
    dtm = dtm.fillna(0)
    dtm = dtm.sort_index(axis=0)

    # Get normalized term frequency (tf) matrix
    tf = dtm.values
    doc_len = tf.sum(axis=1)
    tf = np.divide(tf, doc_len[:, None])

    # Get idf
    df = np.where(tf > 0, 1, 0)
    idf = np.log(np.divide(len(docs), np.sum(df, axis=0))) + 1
    smoothed_idf = np.log(np.divide(len(docs) + 1, np.sum(df, axis=0) + 1)) + 1

    # Get tf-idf
    tf_idf = normalize(tf * idf)
    smoothed_tf_idf = normalize(tf * smoothed_idf)

    # Use the smoothed_tf_idf to compare the distence between target and others
    target = smoothed_tf_idf[doc_id, :]
    score = [1 - distance.cosine(target, smoothed_tf_idf[i, :]) for i in range(len(docs))]
    dtm['score'] = score
    dtm_sort = dtm.sort_values(by=['score'], ascending=False)
    top_sim_index = dtm_sort.index.values[1]
    top_sim_score = dtm_sort['score'].values[1]

    return top_sim_index, top_sim_score


if __name__ == "__main__":

    # Test Q1
    text = "Composed of 3 CDs and quite a few songs (I haven't an exact count), \
          all of which are heart-rendering and impressively remarkable. \
          It has everything for every listener -- from fast-paced and energetic \
          (Dancing the Tokage or Termina Home), to slower and more haunting (Dragon God), \
          to purely beautifully composed (Time's Scar), \
          to even some fantastic vocals (Radical Dreamers).\
          This is one of the best videogame soundtracks out there, \
          and surely Mitsuda's best ever. ^_^"

    tokens = tokenize(text)

    print("Q1 tokens:", tokens)

    # Test Q2

    with open("positive-words.txt", 'r') as f:
        positive_words = [line.strip() for line in f]

    with open("negative-words.txt", 'r') as f:
        negative_words = [line.strip() for line in f]

    acc = performance_evaluate("amazon_review_300.csv",
                               positive_words, negative_words)
    print("\nQ2 accuracy: {0:.2f}".format(acc))

    # Test Q3
    data = pd.read_csv("amazon_review_300.csv")
    # pick any doc id, e.g. 10, 207
    doc_id = 207
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])

    doc_id = 10
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])

    print("\n\n\n\nThe following 7 examples are the evidence to support this function work:\n")
    doc_id = 117
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])

    doc_id = 266
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\n\n\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])

    doc_id = 67
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\n\n\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])

    doc_id = 255
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\n\n\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])

    doc_id = 158
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\n\n\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])

    doc_id = 299
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\n\n\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])

    doc_id = 131
    sim_doc_id, sim = find_similar_doc(data["review"], doc_id)
    print("\n\n\nSimilarity between {0} and {1} is {2:.2f}: "
          .format(doc_id, sim_doc_id, sim))
    print("\nselected doc: ", data.loc[doc_id]["review"])
    print("\nsimilar doc: ", data.loc[sim_doc_id]["review"])
