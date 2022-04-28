from src import chaos, diabetes, dna, facerecognition, finance, iris, recommendation, salaries, titanic, wines
import streamlit as st


# Object to store all the apps
class MultiApp:
    def __init__(self):
        self.apps = []

    # Adds a new application to the list
    # title: title of the app. Appears in the dropdown in the sidebar.
    # function: the python function to render the app.
    def add_app(self, title, function):
        self.apps.append({"title": title,
                          "function": function})

    # Renders a radio menu with all the apps
    # message: message to display above the radio menu
    # returns the function of the selected app
    def menu(self, message):
        current_app = st.radio(message, self.apps, format_func=lambda application: application['title'])
        return current_app['function']


if __name__ == '__main__':
    # General configuration
    st.set_page_config(layout="wide")
    apps = MultiApp()

    # Add all your application here
    # TODO: Add recommendation app
    # TODO: Add face-recognition app
    # TODO: Sentiment analysis on customer support data
    #  - https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
    apps.add_app("Chaos", chaos.app)
    apps.add_app("Finance", finance.app)
    apps.add_app("Salaries", salaries.app)  # TODO: make it more interactive and more methods
    apps.add_app("DNA", dna.app)            # TODO: amino-acid translation, https://www.youtube.com/watch?v=3FQZqS300qE
    apps.add_app("Diabetes", diabetes.app)  # TODO: update this to be more interactive
    apps.add_app("Wines", wines.app)        # TODO: create app for wine classification
    apps.add_app("Titanic", titanic.app)    # TODO: update this to be more interactive and more methods
    apps.add_app("Iris", iris.app)

    # Add the menu to the sidebar
    with st.sidebar:
        st.title("Choose an app")
        app = apps.menu("Apps")

    # Render the selected app
    app()
