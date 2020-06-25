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
    sl.ensembles.AWE(GaussianNB())
    #OurAWE(GaussianNB())
]
clf_names = [
    'SEA',
    'AUE',
    'WAE',
    'AWE'
    #'OurAWE'
]

alfa = .05                  #### PRÓG NIEPEWNOŚCI
scores = []                 #### TABLICA PRZECHOWUJĄCA WYNIKI POSZCZEGÓLNYCH EWALUACJI
streams_names = dict([      #### SŁOWNIK WYKORZYSTYWANY DO ZACHOWANIA CZYTELNOŚCI PREZENTACJI DANYCH
    (0, "STRUMIENI Z DRYFEM NAGŁYM"), 
    (1, "STRUMIENI Z DRYFEM DUALNY"), 
    (2, "STRUMIENI Z DRYFEM INKREMENTALNY")
])
streams_list = [[] for i in range (3)]  #### LISTA TRZECH LIST ZAWIERAJĄCYCH STRUMIENIE DANYCH


""" GENEROWANIE STRUMIENI DANYCH """
for rstate in rand_states:
    stream = sl.streams.StreamGenerator(
        n_chunks = 200,
        chunk_size = 250,
        random_state = rstate,
        n_features = 10,
        n_classes = 2,
        n_drifts = 1
    )
    streams_list[0].append(stream)      #### LISTA STRUMIENI Z DRYFEM NAGŁYM

    stream = sl.streams.StreamGenerator(
        n_chunks = 200,
        chunk_size = 250,
        random_state = rstate,
        n_features = 10,
        n_classes = 2,
        n_drifts = 1,
        concept_sigmoid_spacing = 5
    )
    streams_list[1].append(stream)      #### LISTA STRUMIENI Z DRYFEM DUALNY

    stream = sl.streams.StreamGenerator(
        n_chunks = 200,
        chunk_size = 250,
        random_state = rstate,
        n_features = 10,
        n_classes = 2,
        n_drifts = 1,
        concept_sigmoid_spacing = 5,
        incremental = True
    )
    streams_list[2].append(stream)      #### LISTA STRUMIENI Z DRYFEM INKREMENTALNY


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
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))

    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(avg_mean_scores[i], avg_mean_scores[j])

    """ TWORZENIE TABELI FORMATUJĄCEJ WYNIKI """
    names_column = np.array([["SEA"], ["AUE"], ["WAE"], ["AWE"]])
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
    print("Lepszy znacząco statystycznie:\n", stat_better_table, "\n")

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