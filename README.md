# MSI-AWE
Repozytorium pod projekt z Metod Sztucznej Inteligencji

Tematem projektu jest "Wykorzystanie algorytmu AWE w przetwarzaniu strumieni danych zawierających dryf koncepcji", a także wykorzystanie metod porównawczych: AUE, SEA i WAE w celu oceny jakości oraz wyciągnięcia wniosków na temat skuteczności badanego algorytmu.

# DZIAŁANIE
Program ma na celu przeprowadzenie ewaluacji serii strumieni danych zawierających dryfy koncepcji wykorzystując algorytmy do tego stworzone. Wyliczana jest dokładność predykcji poszczególnych algorytmów z wykorzystaniem 3 metryk, tj. F1_score, G-mean oraz Balanced Accuracy, przeprowadzane są parowe testy statystyczne, aby określić czy wykorzystane metody różnią się od siebie statystycznie.

# STRUMIENIE
Wykorzystane w programie strumienie mają następujące cechy: 
- ilość bloków danych: 200
- rozmiar bloku danych: 250 próbek
- ilość dryfów koncepcji w strumieniu : 3
- liczba cech: 10
- liczba klas: 2

W celu umożliwienia powtarzalności eksperymentu został wykorzystany parametr 'random_state'. W eksperymencie wykorzystano łącznie 30 strumieni, badano zachowanie algorytmów przy 3 rodzajach dryfu koncepcji (dryf nagły, dualny, inkrementalny), po 10 strumieni na dryf. Uzyskane ze strumieni wyniki zostały uśrednione. Zmiany parametrów strumieni można dokonać poprzez edycję zmiennych zadeklarowanych w głównym pliku programu (main.py)

# OUTPUT
Program wypisuje w konsoli poszczególne tabele zawierające wyżej wymienione elementy oraz rysuje wykresy dokładności predykcji wykorzystanych algorytmów w zależności od aktualnego bloku danych (chunk). Każdy z wykresów zawiera 3 podwykresy zawierające wyniki dla poszczególnych metryk. Wykresy te są zapisywane w formacie .PNG w folderze "charts" tworzonym w miejscu uruchomienia programu.
