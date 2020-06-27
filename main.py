import os
import strlearn as sl
import numpy as np
from ourAWE import OurAWE
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from tabulate import tabulate


""" LISTA RANDOM STATE'ÓW WYKORZYSTANYCH DO ODTWARZANIA POWTARZALNYCH EKSPERYMENTÓW """
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
    sl.ensembles.WAE(GaussianNB()),
    sl.ensembles.AWE(GaussianNB()),
    OurAWE(GaussianNB())
]
clf_names = [
    'SEA',
    'AUE',
    'WAE',
    'AWE',
    'OurAWE'
]

""" KONTROLA BUDOWY STRUMIENIA DANYCH """
n_chunks = 200              #### LICZBA BLOKÓW DANYCH
chunk_size = 250            #### ROZMIAR BLOKU DANYCH
n_drifts = 3                #### LICZBA WYSTĄPIENIA DRYFU KONCEPCJI W STRUMIENIU
n_features = 10             #### LICZBA CECH
n_classes = 2               #### LICZBA KLAS PROBLEMU

alfa = .05                  #### PRÓG NIEPEWNOŚCI
scores = []                 #### TABLICA PRZECHOWUJĄCA WYNIKI POSZCZEGÓLNYCH EWALUACJI
streams_names = dict([      #### SŁOWNIK WYKORZYSTYWANY DO ZACHOWANIA CZYTELNOŚCI PREZENTACJI DANYCH
    (0, "STRUMIENI Z DRYFEM NAGŁYM"), 
    (1, "STRUMIENI Z DRYFEM DUALNYM"), 
    (2, "STRUMIENI Z DRYFEM INKREMENTALNYM")
])
streams_list = [[] for i in range (3)]  #### LISTA TRZECH LIST DO PRZECHOWYWANIA STRUMIENI DANYCH


""" GENEROWANIE STRUMIENI DANYCH """
for rstate in rand_states:
    stream = sl.streams.StreamGenerator(
        n_chunks = n_chunks,
        chunk_size = chunk_size,
        random_state = rstate,
        n_features = n_features,
        n_classes = n_classes,
        n_drifts = n_drifts
    )
    streams_list[0].append(stream)      #### LISTA STRUMIENI Z DRYFEM NAGŁYM

    stream = sl.streams.StreamGenerator(
        n_chunks = n_chunks,
        chunk_size = chunk_size,
        random_state = rstate,
        n_features = n_features,
        n_classes = n_classes,
        n_drifts = n_drifts,
        concept_sigmoid_spacing = 5
    )
    streams_list[1].append(stream)      #### LISTA STRUMIENI Z DRYFEM DUALNYM

    stream = sl.streams.StreamGenerator(
        n_chunks = n_chunks,
        chunk_size = chunk_size,
        random_state = rstate,
        n_features = n_features,
        n_classes = n_classes,
        n_drifts = n_drifts,
        concept_sigmoid_spacing = 5,
        incremental = True
    )
    streams_list[2].append(stream)      #### LISTA STRUMIENI Z DRYFEM INKREMENTALNYM


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


""" EWALUACJA STRUMIENI TEGO SAMEGO TYPU """
evaluator = sl.evaluators.TestThenTrain(metrics)
for st_id, stream_ in enumerate(streams_list):
    for st in stream_:
        evaluator = sl.evaluators.TestThenTrain(metrics)
        evaluator.process(st, clfs)
        scores.append(evaluator.scores)

    mean_scores = np.mean(scores, axis = 0)     #### UŚREDNIANIE WYNIKÓW STRUMIENI TEGO SAMEGO TYPU
    scores = []
    avg_mean_scores = []
    for ms in mean_scores:
        avg_mean_scores.append(np.mean(ms, axis = 0))
    
    """ PAROWE TESTY STATYSTYCZNE """
    print("\n -----||| ANALIZA STATYSTYCZNA DLA {} |||-----\n".format(streams_names[st_id]))

    """ TABELA ŚREDNIEJ DOKŁADNOŚCI PREDYKCJI """
    mean_accuracy = [[np.mean(avg_mean_scores[i])] for i in range (len(clfs))]
    names_column = np.array([["SEA"], ["AUE"], ["WAE"], ["AWE"], ["OurAWE"]])
    accuracy_header = ["Algorytm", "F1_score", "G-mean", "Balanced accuracy", "Średnia"]
    mean_accuracy_table = np.concatenate((names_column, avg_mean_scores), axis=1)
    mean_accuracy_table = np.concatenate((mean_accuracy_table, np.array(mean_accuracy)), axis=1)
    mean_accuracy_table = tabulate(mean_accuracy_table, accuracy_header, floatfmt=".2f")
    print("Średnia dokładność predykcji algorytmów dla poszczególnych metryk:\n", mean_accuracy_table, "\n")

    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(avg_mean_scores[i], avg_mean_scores[j])

    """ TWORZENIE TABELI FORMATUJĄCEJ WYNIKI """
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, clf_names, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, clf_names, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table, "\n")

    """ TABELA PRZEWAGI """
    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), clf_names)
    print("Przewaga:\n", advantage_table, "\n")

    """ TABELA RÓŻNIC STATYSTYCZNIE ZNACZĄCYCH """
    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate((names_column, significance), axis=1), clf_names)
    print("Znaczenie statystyczne (alpha = 0.05):\n", significance_table, "\n")

    """ WYNIK KOŃCOWY ANALIZY STATYSTYCZNEJ (ALGORYTMY STATYSTYCZNIE ZNACZĄCO LEPSZE OD POZOSTAŁYCH) """
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), clf_names)
    print("Statystycznie znacząco lepszy:\n", stat_better_table, "\n")

    """ RYSOWANIE WYKRESU WYNIKÓW EWALUACJI """
    fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8), num = "Wykresy dla {}".format(streams_names[st_id].lower()))
    for m, metric in enumerate(metrics):
        ax[m].set_title(metrics_names[m])
        ax[m].set_ylim(0, 1)
        for i, clf in enumerate(clfs):
            ax[m].plot(mean_scores[i, :, m], label=clf_names[i])
        plt.ylabel("Mean scores of metric")
        plt.xlabel("Chunk")
        ax[m].legend()
    if not os.path.isdir("charts"):
        os.mkdir("charts")
    plt.savefig("charts/fig_{}".format(st_id))
plt.show()
print("Aby zakończyć proces zamknij wszystkie okna z wykresami")