import pandas as pd
from os import path
from collections import Counter
from glob import glob
from unicodedata import normalize
import joblib
from nltk import ngrams
from tqdm import tqdm


path_name = r'/data/jbarre/corpus_chapitres_txt/*.txt'

MOTIFS_sentences = joblib.load('/data/jbarre/lemmatization/MOTIFS_AWARDS.pkl')
chapitres_index_sentences = joblib.load('/data/jbarre/lemmatization/main_list_index_stanza_sentences_chapitres.pkl')

best_coefs = pd.read_csv(r'/data/jbarre/lemmatization/BEST_MOTIFS_COEFS.csv', index_col='Unnamed: 0')
best_coefs = best_coefs.T
best_coefs.rename(columns={0: 'coefs'}, inplace=True)
best_coefs.sort_values(by=['coefs'], ascending=False, inplace=True)

non_AWARD = best_coefs[:100]
AWARD = best_coefs[-100:]
selected_coefs = pd.concat([non_AWARD, AWARD], axis=0)


def clean_text(txt):
    txt_res = normalize("NFKD", txt).replace('\xa0', ' ')
    txt_res = txt_res.replace("\\", "").replace('\\xa0', '')
    return txt_res
    
def rolling_group_sentences(sentences, group_size, overlap):
    grouped_sentences = [sentences[i - overlap : i + group_size + overlap] for i in range(overlap, len(sentences) - overlap, group_size)]
    #grouped_sentences = [sentences[i:i + group_size] for i in range(0, len(sentences) - overlap + 1, overlap)]
    return grouped_sentences
    
    
def ngram_list(sentences, n_gram_len):
    # Flatten the list of sentences and split into words
    words = [word for sentence in sentences for word in sentence.split()]
    # Generate n-grams
    n_grams = ngrams(words, n_gram_len)
    return n_grams
    
def generate_n_grams_dico(n):
    dico_list_ngrams = {}
    for i in range(1, n+1):
        list_name = f"list_{i}_gram"
        dico_list_ngrams[list_name]=[]

    return dico_list_ngrams
    
    
    
def get_ngrams(chunk, len_ngrams=5):
    DICT_TEMP = generate_n_grams_dico(len_ngrams)#key = "list_{i}_gram" avec list vide   
    
    for i in range(1, len_ngrams+1):
            DICT_TEMP[f"list_{i}_gram"].extend(ngram_list(chunk, i))# too big for memory ?
            
    return DICT_TEMP   
    
    
def compute_score(list_chunks, df_coef, len_ngrams):
    list_scores = []
    for chunk in tqdm(list_chunks):
        score = 0
        DICT_NGRAMS = get_ngrams(chunk, len_ngrams)
        
        for i in range(1, len_ngrams+1):
            list_courante = DICT_NGRAMS[f"list_{i}_gram"]
            for ngram in list_courante:
                ngram_str = '_'.join(ngram)
                if ngram_str in list(df_coef.index):
                    temp_score = df_coef.loc[df_coef.index == ngram_str]
                    score += temp_score.values[0][0]
        list_scores.append(score)
    return list_scores
    
    
    
def signal_canon(path_name, list_index, list_motifs, df_coef, group_size, overlap, len_ngrams=5):# 100, 10
    
    df_main = pd.DataFrame()

    for doc in tqdm(glob(path_name)):
        
        doc_name = path.splitext(path.basename(doc))[0]
        print(doc_name)
        if doc_name in list_index:
            index_of_element = list_index.index(doc_name)
            list_chunks = rolling_group_sentences(list_motifs[index_of_element], group_size, overlap)
            list_scores = compute_score(list_chunks, df_coef, len_ngrams)
            print(pd.Series(list_scores).mean())
            df_main[doc_name] = pd.Series(list_scores)
        else:
            pass
    
    return df_main.fillna(0)  
    
        
df_scores = signal_canon(path_name, chapitres_index_sentences, MOTIFS_sentences, selected_coefs, 30, 10)#selected_coefs/best_coefs
df_scores.to_csv("/data/jbarre/signal_motifs_chapitres.csv")

