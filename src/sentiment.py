import streamlit as st
import os

# Get directory paths
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, "data/{}")


def app():
    st.title("Work in progress")


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
