import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB


# Q1


def classify(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Use pipline and gridsearch to find the best parameters
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC())])
    parameters = {'tfidf__min_df': [1, 2, 3],
                  'tfidf__stop_words': [None, "english"],
                  'clf__C': [0.5, 1, 5],
                  }

    metric = "f1_macro"
    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=6)
    gs_clf = gs_clf.fit(train["text"], train["label"])
    dic_best_para = gs_clf.best_params_
    # Print out the best parameters and best f1 score
    for param_name in dic_best_para:
        print(param_name, ": ", gs_clf.best_params_[param_name])
    print("best f1 score:", gs_clf.best_score_)

    # Use the best paramater to built the object
    tfidf_vect = TfidfVectorizer(
        min_df=dic_best_para['tfidf__min_df'], stop_words=dic_best_para['tfidf__stop_words'])
    SVC = LinearSVC(penalty='l2', C=dic_best_para['clf__C'])
    # Generate tfidf matrix
    dtm_train = tfidf_vect.fit_transform(train["text"])
    dtm_test = tfidf_vect.transform(test["text"])
    # Train a Linear SVM with all samples
    x_train = dtm_train
    y_train = train["label"]
    x_test = dtm_test
    y_test = test['label']
    SVC.fit(x_train, y_train)
    predicted = SVC.predict(x_test)
    y_score = SVC.decision_function(x_test)
    labels = [str(i) for i in sorted(train["label"].unique())]
    print(classification_report(y_test, predicted, target_names=labels))

    # Plot the ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
    # Calculate auc
    AUC = auc(fpr, tpr)
    print('AUC: %.3f' % AUC)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC of Linear SVM Model')
    plt.show()
    # Calculate average precision
    avg_pre = average_precision_score(y_test, y_score)
    print('Average Precision: %.3f' % avg_pre)
    # Plot the precision_recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score, pos_label=1)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision_Recall_Curve of Linear SVM Model')
    plt.show()
    return None


# Q2


def K_fold_CV(train_file):
    find_k_data = pd.read_csv(train_file)
    tfidf_vect = TfidfVectorizer()
    dtm_find_k = tfidf_vect.fit_transform(find_k_data["text"])
    target = find_k_data['label']
    auc_MultinomialNB = []
    auc_LinearSVC = []
    k_range = np.arange(2, 21)
    for i in k_range:
        clf_1 = MultinomialNB()
        clf_2 = LinearSVC()
        cv_1 = cross_validate(clf_1, dtm_find_k, target, scoring='roc_auc',
                              cv=i, return_train_score=True)
        cv_2 = cross_validate(clf_2, dtm_find_k, target, scoring='roc_auc',
                              cv=i, return_train_score=True)
        auc_avg_1 = np.mean(cv_1['test_score'])
        auc_avg_2 = np.mean(cv_2['test_score'])
        auc_MultinomialNB.append(auc_avg_1)
        auc_LinearSVC.append(auc_avg_2)

    plt.plot(k_range, auc_LinearSVC, label="auc_svm")
    plt.plot(k_range, auc_MultinomialNB, label='auc_nb')
    plt.legend(loc='lower right')
    plt.xlim([2, 20])
    plt.xlabel('K')
    plt.ylabel('AUC')
    plt.show()
    return None


# Q3
def stacking(train_file, test_file):
    Ens_train = pd.read_csv(train_file)
    Ens_test = pd.read_csv(test_file)

    Ens_tfidf = TfidfVectorizer()
    Ens_dtm_train = Ens_tfidf.fit_transform(Ens_train["text"])
    Ens_dtm_test = Ens_tfidf.transform(Ens_test["text"])
    y = Ens_train['label']
    y_test = Ens_test['label']

    test_pred_1 = np.empty((0, 1), float)
    train_pred_1 = np.empty((0, 1), float)
    test_pred_2 = np.empty((0, 1), float)
    train_pred_2 = np.empty((0, 1), float)
    folds = StratifiedKFold(n_splits=6, random_state=1)
    model1 = LinearSVC(random_state=1)
    model2 = GaussianNB()

    for train_indices, val_indices in folds.split(Ens_dtm_train, y.values):
        x_train, x_val = Ens_dtm_train[train_indices], Ens_dtm_train[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        model1.fit(X=x_train, y=y_train)
        train_pred_1 = np.append(train_pred_1, model1.predict(x_val))
        test_pred_1 = np.append(test_pred_1, model1.predict(Ens_dtm_test))

        model2.fit(X=x_train.toarray(), y=y_train)
        train_pred_2 = np.append(train_pred_2, model2.predict(x_val.toarray()))
        test_pred_2 = np.append(test_pred_2, model2.predict(Ens_dtm_test.toarray()))

    gg = y_test
    for i in np.arange(5):
        y_test = np.append(y_test, gg)

    test_pred1, train_pred1 = test_pred_1.reshape(-1, 1), train_pred_1
    train_pred1 = pd.DataFrame(train_pred1)
    test_pred1 = pd.DataFrame(test_pred1)

    test_pred2, train_pred2 = test_pred_2.reshape(-1, 1), train_pred_2
    train_pred2 = pd.DataFrame(train_pred2)
    test_pred2 = pd.DataFrame(test_pred2)
    df = pd.concat([train_pred1, train_pred2], axis=1)
    df_test = pd.concat([test_pred1, test_pred2], axis=1)
    model = DecisionTreeClassifier()
    model.fit(df, y)
    pre = model.predict(np.nan_to_num(df_test))
    labels = [str(i) for i in sorted(Ens_train["label"].unique())]
    print(classification_report(y_test, pre, target_names=labels))

    return None


if __name__ == "__main__":
    # Question 1
    print("Q1")
    classify("assign4_train.csv", "assign4_test.csv")

    # Test Q2
    print("\nQ2")
    K_fold_CV("assign4_train.csv")

    # Test Q3
    print("\nQ3")
    stacking("assign4_train.csv", "assign4_test.csv")
