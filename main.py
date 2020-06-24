import strlearn as sl
import numpy as np
from ourAWE import OurAWE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


rand_states = [
    1111,
    1234,
    1337,
    1410,
    2468,
    5555,
    6996,
    7777,
    8800,
    9999
]


""" LISTA KLASYFIKATORÓW I ICH NAZWY """

clfs = [
    sl.ensembles.SEA(GaussianNB()),
    sl.ensembles.AUE(GaussianNB()),
    sl.ensembles.AWE(GaussianNB()),
    sl.ensembles.WAE(GaussianNB()),
    #OurAWE(GaussianNB())
]
clf_names = [
    'SEA',
    'AUE',
    'AWE',
    'WAE',
    #'OurAWE'
]

""" TESTOWE STRUMIENIE DANYCH """
"""
stream1 = sl.streams.StreamGenerator(
    n_chunks = 10,                 #### ILOŚĆ BLOKÓW DANYCH
    chunk_size = 50,               #### ROZMIAR BLOKU
    random_state = 12345,           #### ZIARNO LOSOWOŚCI
    n_features = 10,                #### ILOŚĆ CECH
    n_classes = 2,                  #### ILOŚĆ KLAS CECHY
    n_drifts = 1                    #### SAMO n_drifts = 1 OZNACZA DRYF NAGŁY
    # concept_sigmoid_spacing = 5   #### + n_drifts = 1 DAJE DRYF SKOKOWY/STOPNIOWY
    # incremental = True            #### + DWA POWYŻSZE DAJE DRYF NARASTAJĄCY
)

stream2 = sl.streams.StreamGenerator(
    n_chunks = 10,
    chunk_size = 50,
    random_state = 12346,
    n_features = 10,
    n_classes = 2,
    n_drifts = 1
)
"""

""" GENEROWANIE STRUMIENI DANYCH """
streams_list = [[] for i in range (3)]  #### LISTA TRZECH LIST ZAWIERAJĄCYCH STRUMIENIE DANYCH

for rstate in rand_states:
    stream = sl.streams.StreamGenerator(
        n_chunks = 20,
        chunk_size = 100,
        random_state = rstate,
        n_features = 10,
        n_classes = 2,
        n_drifts = 1
    )
    streams_list[0].append(stream)      #### LISTA STRUMIENI Z NAGŁYM DRYFEM

    stream = sl.streams.StreamGenerator(
        n_chunks = 20,
        chunk_size = 100,
        random_state = rstate,
        n_features = 10,
        n_classes = 2,
        n_drifts = 1,
        concept_sigmoid_spacing = 5
    )
    streams_list[1].append(stream)      #### LISTA STRUMIENI Z DRYFEM SKOKOWYM

    stream = sl.streams.StreamGenerator(
        n_chunks = 20,
        chunk_size = 100,
        random_state = rstate,
        n_features = 10,
        n_classes = 2,
        n_drifts = 1,
        concept_sigmoid_spacing = 5,
        incremental = True
    )
    streams_list[2].append(stream)      #### LISTA STRUMIENI Z DRYFEM NARASTAJĄCYM


""" WYKORZYSTYWANE METRYKI """
metrics = [
    sl.metrics.f1_score,
    sl.metrics.geometric_mean_score_1,
    sl.metrics.balanced_accuracy_score
]

metrics_names = [
    'F1_score',
    'G-mean',
    'Balanced accuracy'
]

scores = []     #### TABLICA PRZECHOWUJĄCA WYNIKI POSZCZEGÓLNYCH EWALUACJI

""" TESTOWE EWALUACJE
evaluator.process(stream1, clfs)

print("Evaluator.scores: \n", evaluator.scores, "\n-----------------------------------")

scores.append(evaluator.scores)

evaluator.process(stream2, clfs)

print("Evaluator.scores2: \n", evaluator.scores, "\n----------------------------------")

scores.append(evaluator.scores)
mean_scores = np.mean(scores, axis = 0)
print("Mean scores list:\n", mean_scores)
print("\nMean scores first column:\n", mean_scores[:, 0, 2])
print("\n\n\n", evaluator.scores[0, :, 0])
"""

""" EWALUACJA STRUMIENI TEGO SAMEGO TYPU """
evaluator = sl.evaluators.TestThenTrain(metrics)
for stream_ in streams_list:
    for st in stream_:
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(st, clfs)
        scores.append(evaluator.scores)

    mean_scores = np.mean(scores, axis = 0)     #### UŚREDNIANIE WYNIKÓW STRUMIENI TEGO SAMEGO TYPU
    scores = []

    """ RYSOWANIE WYKRESU WYNIKÓW EWALUACJI """
    fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
    for m, metric in enumerate(metrics):
        ax[m].set_title(metrics_names[m])
        ax[m].set_ylim(0, 1)
        for i, clf in enumerate(clfs):
            ax[m].plot(mean_scores[i, :, m], label=clf_names[i])
        plt.ylabel("Mean scores of metric")
        plt.xlabel("Chunk")
        ax[m].legend()
plt.show()

