import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# Get directory paths
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, "data/{}")


@st.cache
def load_data(directory):
    # Read and process the data
    dataset = pd.read_csv(directory.format('diabetes.csv'))
    feature_cols = list(dataset.columns)
    feature_cols.remove('Outcome')
    x_data = dataset[feature_cols]
    y_data = dataset['Outcome']
    return x_data, y_data


# Displays the confusion matrix and metrics associated with the model
def display_prediction(model_name, target, prediction, labels):
    # Sub-header
    st.subheader(model_name)

    # Plot the confusion matrix
    confusion_matrix = metrics.confusion_matrix(target, prediction)
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="Blues_r", fmt='g')
    ax.title.set_text('Confusion matrix')
    ax.set_ylabel('Target label')
    ax.set_xlabel('Predicted label')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    st.pyplot(fig)

    # Metrics
    st.write('Accuracy: ', metrics.accuracy_score(target, prediction))
    st.write('Precision: ', metrics.precision_score(target, prediction))
    st.write('Recall: ', metrics.recall_score(target, prediction))
    st.write('F1 Score: ', metrics.f1_score(target, prediction))
    st.write('AUC: ', metrics.roc_auc_score(target, prediction))


def app():
    # Load data
    st.title('Diabetes Classification App')
    x, y = load_data(data_dir)
    labels = ['No Diabetes', 'Diabetes']

    # Explore data
    st.header('Explore data')
    column = st.selectbox('Select a variable to explore', x.columns)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(labels[0])
        st.bar_chart(x[y == 0][column].value_counts())

    with col2:
        st.subheader(labels[1])
        st.bar_chart(x[y == 1][column].value_counts())

    # Train models
    st.header('Model comparison')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        k_means_model = KMeans(n_clusters=2)
        k_means_model.fit(x_train)
        k_means_prediction = k_means_model.predict(x_test)
        display_prediction('K-Means', y_test, k_means_prediction, labels)

    with col2:
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(x_train, y_train)
        random_forest_prediction = random_forest_model.predict(x_test)
        display_prediction('Random Forest', y_test, random_forest_prediction, labels)

    with col3:
        logistic_regression_model = LogisticRegression(max_iter=1000)
        logistic_regression_model.fit(x_train, y_train)
        logistic_regression_prediction = logistic_regression_model.predict(x_test)
        display_prediction('L. Regression', y_test, logistic_regression_prediction, labels)

    with col4:
        decision_tree_model = DecisionTreeClassifier()
        decision_tree_model.fit(x_train, y_train)
        decision_tree_prediction = decision_tree_model.predict(x_test)
        display_prediction('Decision Tree', y_test, decision_tree_prediction, labels)

    # Predictions
    st.header('Prediction')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('User data')
        user_data = {}
        for i in x.columns:
            user_data[i] = st.number_input(i, value=x[i].mean())

    with col2:
        st.subheader('Predictions')
        user_dataframe = pd.DataFrame(user_data, index=[0])
        st.write('K-Means: ')
        st.info(labels[k_means_model.predict(user_dataframe)[0]])
        st.write('Random Forest: ')
        st.info(labels[random_forest_model.predict(user_dataframe)[0]])
        st.write('L. Regression: ')
        st.info(labels[logistic_regression_model.predict(user_dataframe)[0]])
        st.write('Decision Tree: ')
        st.info(labels[decision_tree_model.predict(user_dataframe)[0]])


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
