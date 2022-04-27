import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import os

# Get directory paths
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, "data/titanic/{}")


def app():
    # Load the data
    train_data = pd.read_csv(data_dir.format('titanic-train.csv'))
    test_data = pd.read_csv(data_dir.format('titanic-test.csv'))

    # Function to preprocess data
    def preprocess_data(data, drop_survived_column=True):
        predictors = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
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

    # Create encoded data for training and testing
    encoded_train = preprocess_data(train_data)
    encoded_test = preprocess_data(test_data, False)

    # Create train and test sets
    x = encoded_train.values
    y = train_data['Survived'].values
    x_test = encoded_test.values

    # Create and train the model
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=0)
    tree_one = DecisionTreeClassifier()
    tree_one.fit(x_train, y_train)

    # Evaluate the model
    tree_accuracy = tree_one.score(x_validation, y_validation)
    st.write("Accuracy: ", tree_accuracy)

    # Make predictions over the test set
    predictions = tree_one.predict(x_test)
    test_data['Survived'] = predictions
    st.write(test_data[['PassengerId', 'Survived']])

    # Create CSV file for submission
    # test_data[['PassengerId', 'Survived']].to_csv('../../predictions/titanic_predictions.csv', index=False)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
