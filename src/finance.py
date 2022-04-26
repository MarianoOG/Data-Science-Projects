import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import pandas as pd
import base64


# Plot Closing Price of Query Symbol
def price_plot(symbol, data):
    df = pd.DataFrame(data[symbol].Close)
    df['Date'] = df.index
    plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
    plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
    plt.xticks(rotation=90)
    plt.title(symbol, fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Closing Price', fontweight='bold')
    st.pyplot()


# Web scraping of S&P 500 data
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df


# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href


def app():
    st.title('S&P 500 App')

    st.markdown("""
    This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price**!
    * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
    * **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
    """)

    st.sidebar.header('User Input Features')

    df = load_data()

    # Sidebar - Sector selection
    sorted_sector_unique = sorted(df['GICS Sector'].unique())
    selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

    # Filtering data
    df_selected_sector = df[(df['GICS Sector'].isin(selected_sector))]

    st.header('Display Companies in Selected Sector')
    st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' +
             str(df_selected_sector.shape[1]) + ' columns.')
    st.dataframe(df_selected_sector)

    st.markdown(file_download(df_selected_sector), unsafe_allow_html=True)

    # https://pypi.org/project/yfinance/

    data = yf.download(tickers=list(df_selected_sector[:10].Symbol),
                       period="ytd",
                       interval="1d",
                       group_by='ticker',
                       auto_adjust=True,
                       prepost=True,
                       threads=True,
                       proxy=None)

    num_company = st.sidebar.slider('Number of Companies', 1, 5)

    if st.button('Show Plots'):
        st.header('Stock Closing Price')
        for i in list(df_selected_sector.Symbol)[:num_company]:
            price_plot(i, data)

    # Get ticker and data on the ticker
    st.sidebar.title("Choose a Ticker")
    ticker_symbol = st.sidebar.selectbox("Ticker", ('GOOGL', 'FB'))
    ticker_data = yf.Ticker(ticker_symbol)

    # ticker_df: Open, High, Low, Close, Volume, Dividends, Stock, Splits
    ticker_df = ticker_data.history(period='1d', start='2010-5-31', end='2020-5-31')

    # Title of the page
    st.title("Simple Stock Price App")
    st.write("Shown are the stock closing price and volume of " + ticker_symbol + "!")

    # Closing Graph
    st.write("## Closing Price")
    st.line_chart(ticker_df.Close)

    # Volume Graph
    st.write("## Volume")
    st.line_chart(ticker_df.Volume)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
