import numpy as np
import pandas as pd
import joblib
import argparse
from tqdm import tqdm
from nltk import ngrams
from collections import Counter

def ngram_frequencies(sentences, n_gram_len, m_most_common):
    # Flatten the list of sentences and split into words
    words = [word for sentence in sentences for word in sentence.split()]
    # Generate n-grams
    n_grams = ngrams(words, n_gram_len)
    # Count n-grams
    ngram_counts = Counter(n_grams)
    # Calculate relative frequencies
    total_ngrams = sum(ngram_counts.values())
    return {'_'.join(list(ngram)): count / total_ngrams for ngram, count in dict(ngram_counts.most_common(m_most_common)).items()}

def moulinette(chapitres_index_sentences, chapitres_lemmas_sentences, len_ngrams=6, m_most_common=2000):

    data = {}

    # Process each novel
    for novel_name, sentences in tqdm(zip(chapitres_index_sentences, chapitres_lemmas_sentences), total=len(chapitres_index_sentences)):
        # Initialize a dictionary for this novel
        novel_data = {}
        # Calculate n-gram frequencies for n=1 to n=len_ngrams
        for n in range(1, len_ngrams):
            novel_data.update(ngram_frequencies(sentences, n, m_most_common))
        # Add the novel data to the main dictionary
        data[novel_name] = novel_data

    # Create the DataFrame
    df_ngram_freq = pd.DataFrame.from_dict(data, orient='index')

    # Replace NaN with 0 (for n-grams that do not appear in some novels)
    df_ngram_freq.fillna(0, inplace=True)

    # Display the DataFrame
    return df_ngram_freq

def post_process(df, len_max):
    len_ngrams = 1
    conteur_temp = 0
    good_cols = []
    for col in list(df.columns):
        if len(col.split('_')) == len_ngrams and conteur_temp < len_max:
            conteur_temp+=1
            good_cols.append(col)
        elif conteur_temp >= len_max:
            len_ngrams+=1
            conteur_temp=0

    return pd.DataFrame(df, columns=good_cols)

if __name__ == '__main__':

    #prendre les arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--n_len_ngrams', help='N len max ngrams', default=6, type=int
    )
    parser.add_argument(
        '-m', '--m_most_common_novels', help='M most common ngrams to extract in each novels', default=2000, type=int
    )    
    parser.add_argument(
        '-o', '--o_most_common_df_final', help='O most common ngrams in df_final', default=1000, type=int
    )

    args = vars(parser.parse_args())

    N = args['n_len_ngrams']
    M = args['m_most_common_novels']
    O = args['o_most_common_df_final']

    print("LOAD CHAPITRES SENTENCES")
    chapitres_lemmas_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_lemmas_stanza_sentences_chapitres.pkl')    
    chapitres_tokens_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_tokens_stanza_sentences_chapitres.pkl')
    chapitres_pos_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_pos_stanza_sentences_chapitres.pkl')
    chapitres_index_sentences = joblib.load('/data/jbarre/lemmatization/main_list_index_stanza_sentences_chapitres.pkl')

    print("GET NGRAMS")
    df_res =  moulinette(chapitres_index_sentences, chapitres_lemmas_sentences, N, M)

    print("SAVE DATAFRAME")

    df_final = post_process(df_res, O)
    df_final.to_csv("AWARDS_ngrams_"+str(O)+".csv")

    df_res.to_csv("AWARDS_ngrams_ALL.csv")

