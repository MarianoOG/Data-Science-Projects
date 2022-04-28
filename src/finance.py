import streamlit as st
import yfinance as yf
import pandas as pd


# Web scraping of S&P 500 data
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    df = df.drop(['SEC filings', 'CIK', 'Headquarters Location'], axis=1)
    df['Founded'] = df['Founded'].apply(lambda x: x[:4])
    df['Founded'] = df['Founded'].astype('int64')
    df['Date first added'] = df['Date first added'].apply(lambda x: x[:10] if type(x) == str else x)
    df['Date first added'] = pd.to_datetime(df['Date first added'] if type(df['Date first added']) == str
                                            else df['Date first added'], format='%Y-%m-%d')
    df['Date first added'] = df['Date first added'].apply(lambda x: x.date())
    df.rename(columns={'Security': 'Company', 'Symbol': 'Ticker'}, inplace=True)
    return df


def app():
    # Title and description
    st.title('S&P 500 Financial App')
    st.markdown("""
    This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock information** \
    (from Yahoo Finance).
    * **Data source 1:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
    * **Data source 2:** [Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC).
    """)

    # Get the data
    df = load_data()

    # Filter data
    st.header('Filter data')
    col1, col2 = st.columns((2, 1))

    # Select filter by sector
    with col1:
        sorted_sector_unique = sorted(df['GICS Sector'].unique())
        selected_sector = st.multiselect('GICS Sector', sorted_sector_unique, sorted_sector_unique)

    # Select filter by founded year
    with col2:
        min_year = int(df['Founded'].min())
        max_year = int(df['Founded'].max())
        selected_year = st.slider('Founded year', min_year, max_year, (min_year, max_year))

    # Filter data
    df_filtered = df[(df['GICS Sector'].isin(selected_sector))]
    df_filtered = df_filtered[(df_filtered['Founded'] >= selected_year[0]) &
                              (df_filtered['Founded'] <= selected_year[1])]

    # Visualize information from filtered data
    st.write('Number of filtered companies: ', df_filtered.shape[0])
    st.dataframe(df_filtered)

    # Visualize filtered data by sector and year
    st.header('Ticker and company stock information')
    if df_filtered.shape[0] == 0:
        st.write('No data to display')
        return None
    col1, col2 = st.columns((1, 1))

    # Get data by company name or ticker
    with col1:
        get_data_by = st.radio('Get data by', ('Company', 'Ticker'))

    # Select company name or ticker
    with col2:
        if get_data_by == 'Company':
            sorted_company_unique = sorted(df_filtered['Company'].unique())
            selected_company = st.selectbox('Company', sorted_company_unique)
            selected_ticker = df_filtered[df_filtered['Company'] == selected_company]['Ticker'].values[0]
        else:
            sorted_ticker_unique = sorted(df['Ticker'].unique())
            selected_ticker = st.selectbox('Ticker', sorted_ticker_unique)

    st.write(df_filtered[df_filtered['Ticker'] == selected_ticker])

    ticker_df = yf.download(tickers=[selected_ticker],
                            period="ytd",
                            interval="1d",
                            group_by='ticker',
                            auto_adjust=True,
                            prepost=True,
                            threads=True,
                            proxy=None)

    # Closing Graph
    st.subheader("Closing Price")
    st.line_chart(ticker_df.drop(['Volume'], axis=1))

    # Volume Graph
    st.subheader("Volume")
    st.line_chart(ticker_df.Volume)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
