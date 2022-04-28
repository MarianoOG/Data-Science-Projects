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

    st.title('Diabetes Classification App')
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write('Accuracy: ', metrics.accuracy_score(y_test, y_prediction))
        st.write('Precision: ', metrics.precision_score(y_test, y_prediction))
        st.write('Recall: ', metrics.recall_score(y_test, y_prediction))
        st.write('F1 Score: ', metrics.f1_score(y_test, y_prediction))
        st.write('AUC: ', metrics.roc_auc_score(y_test, y_prediction))

    with col2:
        # Plot the confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="Blues_r", fmt='g')
        ax.title.set_text('Confusion matrix')
        ax.set_ylabel('Actual label')
        ax.set_xlabel('Predicted label')
        ax.xaxis.set_ticklabels(['No Diabetes', 'Diabetes'])
        ax.yaxis.set_ticklabels(['No Diabetes', 'Diabetes'])
        st.pyplot(fig)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
