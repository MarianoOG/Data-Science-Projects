import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import os

# Get directory paths
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, "data/{}")


def app():
    # Read the data
    dataset = pd.read_csv(data_dir.format('diabetes.csv'))

    feature_cols = list(dataset.columns)
    feature_cols.remove('Outcome')
    x = dataset[feature_cols]
    y = dataset['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_train, y_train)
    y_prediction = log_reg.predict(x_test)

    confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)

    class_names = ['No Diabetes', 'Diabetes']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="Blues_r", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    print('Accuracy: ', metrics.accuracy_score(y_test, y_prediction))
    print('Precision: ', metrics.precision_score(y_test, y_prediction))
    print('Recall: ', metrics.recall_score(y_test, y_prediction))
    print('F1 Score: ', metrics.f1_score(y_test, y_prediction))
    print('AUC: ', metrics.roc_auc_score(y_test, y_prediction))


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
