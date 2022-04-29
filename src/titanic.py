import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import streamlit as st
import os

# Get directory paths
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, "data/titanic/{}")


@st.cache
def load_data():
    # Load the data
    train_data = pd.read_csv(data_dir.format('titanic-train.csv'))
    test_data = pd.read_csv(data_dir.format('titanic-test.csv'))
    target = train_data['Survived']

    # Create encoded data for training and testing
    e_train = preprocess_data(train_data)
    e_test = preprocess_data(test_data, False)

    return e_train, e_test, target


# Function to preprocess data
def preprocess_data(data, drop_survived_column=True):
    predictors = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    if drop_survived_column:
        predictors = predictors.drop(['Survived'], axis=1)
    predictors['Age'] = predictors['Age'].fillna(predictors['Age'].median())
    predictors['Fare'] = predictors['Fare'].fillna(predictors['Fare'].median())
    predictors['Embarked'] = predictors['Embarked'].fillna('S')
    categorical = [cname for cname in predictors.columns if predictors[cname].dtype == "object"]
    numerical = [cname for cname in predictors.columns if
                 predictors[cname].dtype in ['int64', 'float64', 'int32', 'float32']]
    predictors = predictors[categorical + numerical]
    encoded = pd.get_dummies(predictors)
    return encoded


def display_model_results(title, model, validation_data, target, test_data):
    model_validation = model.predict(validation_data)
    model_prediction = model.predict(test_data)
    df_predictions = test_data.copy(deep=True)
    df_predictions['Survived'] = model_prediction
    df_predictions = df_predictions[['PassengerId', 'Survived']]

    st.subheader(title)
    st.write('Accuracy:', accuracy_score(target, model_validation))

    st.write('Results:')
    st.write(df_predictions)
    st.download_button(label="Download data as CSV",
                       data=df_predictions.to_csv(index=False).encode('utf-8'),
                       file_name=title + '_test.csv',
                       mime='text/csv')


def app():
    # Load the data
    encoded_train, encoded_test, target = load_data()
    labels = ['Survived', 'Not Survived']

    # Explore data
    st.header('Explore encoded data')
    st.write(encoded_train)
    column = st.selectbox('Select a variable to explore', encoded_train.drop(['PassengerId'], axis=1).columns)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(labels[0])
        st.bar_chart(encoded_train[target == 0][column].value_counts())

    with col2:
        st.subheader(labels[1])
        st.bar_chart(encoded_train[target == 1][column].value_counts())

    # Train the model
    st.header('Train the model')

    # Create and train the model
    x_train, x_validation, y_train, y_validation = train_test_split(encoded_train.values, target.values,
                                                                    test_size=0.15, random_state=0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        k_means_model = KMeans(n_clusters=2)
        k_means_model.fit(x_train)
        display_model_results('K-Means', k_means_model, x_validation, y_validation, encoded_test)

    with col2:
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(x_train, y_train)
        display_model_results('Random Forest', random_forest_model,
                              x_validation, y_validation, encoded_test)

    with col3:
        logistic_regression_model = LogisticRegression(max_iter=1000)
        logistic_regression_model.fit(x_train, y_train)
        display_model_results('Logistic regression', logistic_regression_model,
                              x_validation, y_validation, encoded_test)

    with col4:
        decision_tree_model = DecisionTreeClassifier()
        decision_tree_model.fit(x_train, y_train)
        display_model_results('Decision Tree', decision_tree_model,
                              x_validation, y_validation, encoded_test)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
