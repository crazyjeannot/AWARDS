import numpy as np
import pandas as pd
import joblib
import pickle
from tqdm import tqdm

def motif_filter(pos_tag, lemma):
    if lemma in verbs_to_keep:
        return lemma
    elif pos_tag in pos_to_lemma:
        return lemma
    else:#elif pos_tag in pos_to_keep
        return pos_tag



def get_motifs(chapitres_pos_sentences, chapitres_lemmas_sentences):
    list_motifs_main = []
    for pos_novel, lemmas_novel in tqdm(zip(chapitres_pos_sentences, chapitres_lemmas_sentences), total=len(chapitres_pos_sentences)): #on loop sur les romans
        list_motifs_novel = []
        for pos_sentence, lemmas_sentence in zip(pos_novel, lemmas_novel):# on loop sur les phrases
            list_motifs_sentence = []
            for pos, lemma in zip(pos_sentence.split(), lemmas_sentence.split()): #on loop sur les mots 
                list_motifs_sentence.append(motif_filter(pos, lemma))
            list_motifs_novel.append(' '.join(list_motifs_sentence))
        list_motifs_main.append(list_motifs_novel)
    return list_motifs_main

if __name__ == '__main__':

    verbs_to_keep = ['être', 'avoir', 'aller', 'venir', 'devoir', 'pouvoir', 'falloir']
    pos_to_keep = ['ADJ', 'INTJ', 'NOUN', 'PART', 'PROPN', 'VERB']
    pos_to_lemma = ['ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'PRON', 'SCONJ']

    print("LOAD CHAPITRES SENTENCES")
    chapitres_lemmas_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_lemmas_stanza_sentences_chapitres.pkl')    
    chapitres_tokens_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_tokens_stanza_sentences_chapitres.pkl')
    chapitres_pos_sentences =  joblib.load('/data/jbarre/lemmatization/main_list_pos_stanza_sentences_chapitres.pkl')
    chapitres_index_sentences = joblib.load('/data/jbarre/lemmatization/main_list_index_stanza_sentences_chapitres.pkl')
    
    print("GET MOTIFS")
    list_motifs = get_motifs(chapitres_pos_sentences, chapitres_lemmas_sentences)

    print("SAVE MOTIFS LIST")
    with open('/data/jbarre/lemmatization/MOTIFS_AWARDS.pkl', 'wb') as f1:
        pickle.dump(list_motifs, f1)
