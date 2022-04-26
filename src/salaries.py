import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import os

# Get directory paths
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, "data/{}")


def app():
    # Read the data
    dataset = pd.read_csv(data_dir.format('salarios.csv'))

    # Data preprocessing
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    # Divide the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Linear regression model
    regresor = LinearRegression()
    regresor.fit(x_train, y_train)

    # Gráfica de la regresión lineal simple
    plt.scatter(x_train, y_train, color='blue')
    plt.plot(x_train, regresor.predict(x_train), color='red')
    plt.title('Salario vs Experiencia (Entrenamiento)')
    plt.xlabel('Experiencia')
    plt.ylabel('Salario')
    plt.show()

    # Gráfica de la regresión lineal simple
    plt.scatter(x_test, y_test, color='blue')
    plt.plot(x_test, regresor.predict(x_test), color='red')
    plt.title('Salario vs Experiencia (Prueba)')
    plt.xlabel('Experiencia')
    plt.ylabel('Salario')
    plt.show()

    # Verify the accuracy of the model
    score = regresor.score(x_test, y_test)
    print(score)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
