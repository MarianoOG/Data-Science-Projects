from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import datasets
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


@st.cache(allow_output_mutation=True)
def train_models(x, y):
    # K-Means
    k_means = KMeans(n_clusters=3)
    k_means.fit(x)

    # Random Forest
    random_forest = RandomForestClassifier()
    random_forest.fit(x, y)

    # Logistic Regression
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(x, y)

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x, y)

    return k_means, random_forest, logistic_regression, decision_tree


def app():
    # Load dataset
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=['species'])

    # Create a dataframe with the features and species
    species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    df = pd.concat([x, y], axis=1)
    df['species'] = df['species'].map(species)

    # Show the data
    st.title('Iris Flower Prediction App')
    st.header('Data Exploration')

    col1, col2 = st.columns((2, 1))

    with col1:
        st.write('Descriptive Statistics')
        st.write(df.describe())

    with col2:
        st.write('Distribution of Species')
        st.write(df['species'].value_counts())
        st.write('Download the dataset with labels')
        st.download_button(label="Download data as CSV",
                           data=df.to_csv(index=False).encode('utf-8'),
                           file_name='iris_with_labels.csv',
                           mime='text/csv')

    # User input
    st.header('User Input')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sepal_length = st.slider('Sepal length (cm)',
                                 float(df['sepal length (cm)'].min()),
                                 float(df['sepal length (cm)'].max()),
                                 float(df['sepal length (cm)'].mean()))
    with col2:
        sepal_width = st.slider('Sepal width (cm)',
                                float(df['sepal width (cm)'].min()),
                                float(df['sepal width (cm)'].max()),
                                float(df['sepal width (cm)'].mean()))
    with col3:
        petal_length = st.slider('Petal length (cm)',
                                 float(df['petal length (cm)'].min()),
                                 float(df['petal length (cm)'].max()),
                                 float(df['petal length (cm)'].mean()))
    with col4:
        petal_width = st.slider('Petal width (cm)',
                                float(df['petal width (cm)'].min()),
                                float(df['petal width (cm)'].max()),
                                float(df['petal width (cm)'].mean()))

    # Make dataframe with user input
    user_data = {'sepal length (cm)': sepal_length,
                 'sepal width (cm)': sepal_width,
                 'petal length (cm)': petal_length,
                 'petal width (cm)': petal_width}
    user_features = pd.DataFrame(user_data, index=[0])
    user_data['species'] = 'Your input'
    species[4] = 'Your input'
    df = df.append(user_data, ignore_index=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        for s in species.values():
            temp = df[df['species'] == s]
            ax.scatter(temp['sepal length (cm)'], temp['sepal width (cm)'], label=s)
        ax.set_xlabel('Sepal length (cm)')
        ax.set_ylabel('Sepal width (cm)')
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        for s in species.values():
            temp = df[df['species'] == s]
            ax.scatter(temp['petal length (cm)'], temp['petal width (cm)'], label=s)
        ax.set_xlabel('Petal length (cm)')
        ax.set_ylabel('Petal width (cm)')
        ax.legend()
        st.pyplot(fig)

    # Models and predictions
    st.header('Models and predictions')
    x_train, x_test, y_train, y_test = train_test_split(x, y['species'], test_size=0.2, random_state=0)
    k_means_model, random_forest_model, logistic_regression_model, decision_tree_model = train_models(x_train, y_train)
    col1, col2, col3, col4 = st.columns(4)

    # K-Means
    with col1:
        # Score and prediction
        y_prediction = k_means_model.predict(x_test)
        accuracy = adjusted_rand_score(y_test, y_prediction)
        k_means_prediction = k_means_model.predict(user_features)

        # Write results
        st.subheader('K-Means')
        st.write('Accuracy: ')
        st.write(accuracy)
        st.write('Prediction:')
        st.info(species[k_means_prediction[0]])

    # Random Forest
    with col2:
        # Score and prediction
        score = random_forest_model.score(x_test, y_test)
        random_forest_prediction = random_forest_model.predict(user_features)
        random_forest_prediction_proba = random_forest_model.predict_proba(user_features)

        # Write results
        st.subheader('Random Forest')
        st.write('Accuracy:')
        st.write(score)
        st.write('Prediction:')
        st.info(species[random_forest_prediction[0]])

        st.write('Prediction Probability', random_forest_prediction_proba)

    # Logistic Regression
    with col3:
        # Score and prediction

        # Write results
        st.subheader('Logistic Regression')
        score = logistic_regression_model.score(x_test, y_test)
        st.write('Accuracy: ')
        st.write(score)
        logistic_regression_prediction = logistic_regression_model.predict(user_features)
        st.write('Prediction:')
        st.info(species[logistic_regression_prediction[0]])
        logistic_regression_prediction_proba = logistic_regression_model.predict_proba(user_features)
        st.write('Prediction Probability:', logistic_regression_prediction_proba)

    # Decision Tree
    with col4:
        # Score and prediction
        score = decision_tree_model.score(x_test, y_test)
        decision_tree_prediction = decision_tree_model.predict(user_features)
        decision_tree_prediction_proba = decision_tree_model.predict_proba(user_features)
        
        # Write results
        st.subheader('Decision Tree')
        st.write('Accuracy: ')
        st.write(score)
        st.write('Prediction:')
        st.info(species[decision_tree_prediction[0]])
        st.write('Prediction Probability:', decision_tree_prediction_proba)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
