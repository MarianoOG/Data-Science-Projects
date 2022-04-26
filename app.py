from src import chaos, diabetes, dna, finance, iris, salaries, sentiment, titanic, wines
import streamlit as st


class MultiApp:
    def __init__(self):
        self.apps = []

    # Adds a new application.
    # title: title of the app. Appears in the dropdown in the sidebar.
    # func: the python function to render the app.
    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "func": func
        })

    def menu(self, message):
        current_app = st.radio(message, self.apps, format_func=lambda application: application['title'])
        return current_app['func']


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    apps = MultiApp()

    # Add all your application here
    # TODO: Add recommendation app
    apps.add_app("Chaos Visualizations", chaos.app)
    apps.add_app("Diabetes", diabetes.app)
    apps.add_app("Finance", finance.app)
    apps.add_app("Salaries", salaries.app)
    apps.add_app("Sentiment Analysis", sentiment.app)
    apps.add_app("DNA", dna.app)
    apps.add_app("Titanic", titanic.app)
    apps.add_app("Iris", iris.app)
    apps.add_app("Wines", wines.app)

    with st.sidebar:
        st.title("Choose an app")
        app = apps.menu("Apps")

    app()
