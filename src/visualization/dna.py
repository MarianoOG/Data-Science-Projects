import streamlit as st
import pandas as pd


def dna_nucleotide_count(seq):
    return dict([('A', seq.count('A')),
                ('T', seq.count('T')),
                ('G', seq.count('G')),
                ('C', seq.count('C'))])


def app():

    st.title("DNA Nucleotide Count Web App")
    st.write("This app counts the nucleotide composition of query DNA!")
    st.write("***")

    st.header('Enter DNA sequence')
    sequence_input = ">DNA Query 2\n" \
                     "GAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\n" \
                     "ATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\n" \
                     "TGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"

    sequence = st.text_area("Sequence input", sequence_input, height=250)
    sequence = sequence.splitlines()
    sequence = sequence[1:]  # Skips the sequence name (first line)
    sequence = ''.join(sequence)  # Concatenates list to string

    st.write("***")

    # Prints the input DNA sequence
    st.header('INPUT (DNA Query)')
    st.code(sequence)

    # DNA nucleotide count
    st.header('OUTPUT (DNA Nucleotide Count)')

    col1, col2 = st.columns((1, 1))
    col3, col4 = st.columns((1, 1))

    # 1. Print dictionary
    with col1:
        st.subheader('1. Print dictionary')
        x = dna_nucleotide_count(sequence)
        st.write(x)

    # 2. Print text
    with col2:
        st.subheader('2. Print text')
        st.write('There are  ' + str(x['A']) + ' adenine (A)')
        st.write('There are  ' + str(x['T']) + ' thymine (T)')
        st.write('There are  ' + str(x['G']) + ' guanine (G)')
        st.write('There are  ' + str(x['C']) + ' cytosine (C)')

    # 3. Display DataFrame
    with col3:
        st.subheader('3. Display DataFrame')
        df = pd.DataFrame.from_dict(x, orient='index')
        df = df.rename({0: 'count'}, axis='columns')
        df.reset_index(inplace=True)
        df = df.rename(columns={'index': 'nucleotide'})
        st.write(df)

    # 4. Display Bar Chart using Altair
    with col4:
        st.subheader('4. Display Bar chart')
        # TODO: add bar chart the altair library
        # p = alt.Chart(df).mark_bar().encode( x='nucleotide', y='count')
        # p = p.properties(width=alt.Step(80))
        st.write("chart soon")


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
