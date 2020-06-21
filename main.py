import strlearn as sl
import numpy as np
from ourAWE import OurAWE
from sklearn.naive_bayes import GaussianNB


rstate = 1234

""" GENEROWANIE STRUMIENIA DANYCH """

stream = sl.streams.StreamGenerator(
    n_chunks = 200,         #### ILOŚĆ BLOKÓW DANYCH
    chunk_size = 500,       #### ROZMIAR BLOKU
    random_state = rstate,  #### ZIARNO LOSOWOŚCI
    n_features = 10,        #### ILOŚĆ CECH
    n_classes = 2,          #### ILOŚĆ KLAS CECHY
    n_drifts = 1            #### LICZBA DRYFÓW KONCEPCJI
)


""" LISTA KLASYFIKATORÓW I ICH NAZWY """

clfs = [
    sl.ensembles.SEA(GaussianNB()),
    sl.ensembles.AUE(GaussianNB()),
    sl.ensembles.AWE(GaussianNB()),
    sl.ensembles.WAE(GaussianNB()),
    OurAWE(GaussianNB())
]

clf_names = [
    'SEA',
    'AUE',
    'AWE',
    'WAE',
    'OurAWE'
]