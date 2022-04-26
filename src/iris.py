from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import streamlit as st
import pandas as pd


def k_means():
    iris = datasets.load_iris()

    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=['species'])

    model = KMeans(n_clusters=3)
    model.fit(x)
    y_labels = model.predict(x)
    print('y_labels: ', y_labels)

    accuracy = adjusted_rand_score(y['species'], y_labels)
    print('accuracy: ', accuracy)

    plt.scatter(x['sepal length (cm)'], x['sepal width (cm)'], c=y['species'], cmap='rainbow')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.show()

    plt.scatter(x['petal length (cm)'], x['petal width (cm)'], c=y['species'], cmap='rainbow')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.show()


def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


def app():
    st.title('Data')

    st.write("This is the `Data` page of the multi-page app.")

    st.write("The following is the DataFrame of the `iris` dataset.")

    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='class')
    df = pd.concat([x, y], axis=1)
    df['class'] = df['class'].map({0: "setosa", 1: "versicolor", 2: "virginica"})

    st.write(df)

    st.title('Model')

    st.write('This is the `Model` page of the multi-page app.')

    st.write('The model performance of the Iris dataset is presented below.')

    # Load iris dataset
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Model building
    st.header('Model performance')
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    st.write('Accuracy:')
    st.write(score)

    st.title("Simple Iris Flower Prediction App")
    st.write("This app predicts the **Iris flower** type!")

    st.sidebar.header('User Input Parameters')

    df = user_input_features()

    st.subheader('User Input parameters')
    st.write(df)

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    clf = RandomForestClassifier()
    clf.fit(x, y)

    prediction = clf.predict(df)
    prediction_proba = clf.predict_proba(df)

    st.subheader('Class labels and their corresponding index number')
    st.write(iris.target_names)

    st.subheader('Prediction')
    st.write(iris.target_names[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
