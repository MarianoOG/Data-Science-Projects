from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import streamlit as st
import pandas as pd


def app():
    st.title('Iris Flower Prediction App')

    # Data
    st.header('Data')
    st.write("The following is the DataFrame of the `iris` dataset.")

    # Load dataset
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=['species'])

    # Show the data
    df = pd.concat([x, y], axis=1)
    species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    df['species'] = df['species'].map(species)
    st.write(df)

    # Explore the data
    st.header('Exploration')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Distribution of Species by sepal length and width')
        fig, ax = plt.subplots()
        ax.scatter(x['sepal length (cm)'], x['sepal width (cm)'], c=y['species'], cmap='rainbow')
        ax.set_xlabel('sepal length (cm)')
        ax.set_ylabel('sepal width (cm)')
        st.pyplot(fig)

    with col2:
        st.subheader('Distribution of Species by petal length and width')
        fig, ax = plt.subplots()
        ax.scatter(x['petal length (cm)'], x['petal width (cm)'], c=y['species'], cmap='rainbow')
        ax.set_xlabel('petal length (cm)')
        ax.set_ylabel('petal width (cm)')
        st.pyplot(fig)

    st.header('Models')

    st.subheader('K-Means')
    model = KMeans(n_clusters=3)
    model.fit(x)
    y_labels = model.predict(x)
    accuracy = adjusted_rand_score(y['species'], y_labels)
    st.write('Accuracy: ', accuracy)

    # Model
    st.subheader('Random Forest Classifier Model')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    st.write('Accuracy: ', score)

    # Prediction
    st.header('Predictions')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)

    with col2:
        sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)

    with col3:
        petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)

    with col4:
        petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)

    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}

    user_features = pd.DataFrame(data, index=[0])

    k_means_prediction = model.predict(user_features)
    random_forest_prediction = clf.predict(user_features)
    prediction_proba = clf.predict_proba(user_features)

    st.write('K-means prediction', species[k_means_prediction[0]])
    st.write('Random forest prediction', species[random_forest_prediction[0]])
    st.write('Prediction Probability', prediction_proba)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
