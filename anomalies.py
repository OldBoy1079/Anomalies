import numpy as np
import pandas as pd
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import matplotlib.pyplot as plt




data = np.load('celles.npy', allow_pickle=True)


# Загрузка
with open('ic.pkl', 'rb') as f:
    ic = pickle.load(f)
with open('id.pkl', 'rb') as f:
    id = pickle.load(f)
with open('vc.pkl', 'rb') as f:
    vc = pickle.load(f)
with open('vd.pkl', 'rb') as f:
    vd = pickle.load(f)
with open('tc.pkl', 'rb') as f:
    tc = pickle.load(f)
with open('td.pkl', 'rb') as f:
    td = pickle.load(f)

all_data1 = np.load('all_data1.npy', allow_pickle=True)
all_data2 = np.load('all_data2.npy', allow_pickle=True)
all_data3 = np.load('all_data3.npy', allow_pickle=True)

names1 = np.load('names1.npy', allow_pickle=True)
names2 = np.load('names2.npy', allow_pickle=True)
names3 = np.load('names3.npy', allow_pickle=True)




def prepare_data_for_prefixspan(data):
    """
    Подготавливает данные для алгоритма PrefixSpan, извлекая только идентификаторы событий

    :param data: numpy array с последовательностями событий и дополнительной информацией
    :return: список списков строк - только идентификаторы событий для каждого временного ряда
    """

    prepared_data = []
    for sequence in data:
        # Берем только первый столбец (события)
        events = sequence[:, 0].tolist()
        prepared_data.append(events)
    return prepared_data


def frequence_sequences(all_data1):
    """
    Находит частые последовательности в данных с помощью алгоритма PrefixSpan

    :param all_data1: подготовленные данные для анализа
    :return: список частых последовательностей (паттернов)
    """

    from prefixspan import PrefixSpan
    prepared_data = prepare_data_for_prefixspan(all_data1)
    ps=PrefixSpan(prepared_data)
    tr=ps.topk(5000, closed=True)
    frequent_item=[]
    for i in range(len(tr)):
        frequent_item.append(tr[i][1])
    return frequent_item


frequent_itemsets_low=frequence_sequences(all_data1)
frequent_itemsets_midum=frequence_sequences(all_data2)
frequent_itemsets_high=frequence_sequences(all_data3)


input_step=10
output_step=10


def prepare_data(x):
    """
    Разбивает данные на неперекрывающиеся окна размера 10

    :param x: данные ячейки
    :return: список окон данных
    """

    data=[]
    start=0
    end=input_step
    while (end<=len(x)):
            data.append(x[start:end])
            end=end+output_step
            start=start+output_step
    return data


def compare_graphs(d1,d2):
    '''
    Сравнивает два графика/набора данных с помощью корреляции

    :param d1: первый набор данных
    :param d2: второй набор данных
    :return: True если корреляция >= 0.93, иначе False
    '''

    size=min(len(d1),len(d2))
    corr = np.corrcoef(d1[0:size], d2[0:size])[0, 1]
    return corr>=0.93


def equal(d1,d2):
    '''
    Проверяет полное равенство двух наборов данных

    :param d1: первый набор данных
    :param d2: второй набор данных
    :return: True если все элементы равны, иначе False
    '''

    return d1[0]==d2[0] and d1[1]==d2[1] and  d1[2]==d2[2] and d1[3]==d2[3] and d1[4]==d2[4] and d1[5]==d2[5]


def transform_one(e, ic, id, vc, vd, tc, td, names, chp):
    """
    Преобразует сырые данные одного окна в последовательность идентификаторов событий

    :param e: данные одного окна (DataFrame с измерениями)
    :param ic, id, vc, vd, tc, td: списки уникальных паттернов измерений
    :param names: список уникальных комбинаций параметров
    :param chp: политика заряда (charge policy)
    :return: список идентификаторов событий для данного окна
    """

    data = []
    for j in e["Ic"].index:
        find = False
        for k in range(len(ic)):
            if (compare_graphs(ic[k][0], e["Ic"][j]) and ic[k][1] == chp):
                IC = "ic" + str(k)
                find = True
                break
        if (not find):
            IC = "ic" + str(len(ic) + 1)
        find = False
        for k in range(len(id)):
            if (compare_graphs(id[k][0], e["Id"][j]) and id[k][1] == chp):
                ID = "id" + str(k)
                find = True
                break
        if (not find):
            ID = "id" + str(len(id) + 1)
        find = False
        for k in range(len(vc)):
            if (compare_graphs(vc[k][0], e["Vc"][j]) and vc[k][1] == chp):
                VC = "vc" + str(k)
                find = True
                break
        if (not find):
            VC = "vc" + str(len(vc) + 1)
        find = False
        for k in range(len(vd)):
            if (compare_graphs(vd[k][0], e["Vd"][j]) and vd[k][1] == chp):
                VD = "vd" + str(k)
                find = True
                break
        if (not find):
            VD = "vd" + str(len(vd) + 1)
        find = False
        for k in range(len(tc)):
            if (compare_graphs(tc[k][0], e["Tc"][j]) and tc[k][1] == chp):
                TC = "tc" + str(k)
                find = True
                break
        if (not find):
            TC = "tc" + str(len(tc) + 1)
        find = False
        for k in range(len(td)):
            if (compare_graphs(td[k][0], e["Td"][j]) and td[k][1] == chp):
                TD = "td" + str(k)
                find = True
                break
        if (not find):
            TD = "td" + str(len(td) + 1)

        find = False
        for m in range(len(names)):
            if (equal(names[m], (IC, ID, VC, VD, TC, TD))):
                find = True
                break
        if (find):
            data.append("E" + str(m))
        else:
            data.append("E" + str(len(names) - 1))

    return data


def search(l1,l2):
    """
    Проверяет, содержится ли подсписок l2 в списке l1

    :param l1: основная последовательность, список идентификаторов событий из данных ячейки
    :param l2: искомый паттерн, список идентификаторов событий из частых последовательностей (то, что нашел PrefixSpan)
    :return: True если l2 содержится в l1, иначе False
    """

    sublist_length = len(l2)
    if sublist_length == 0:
        return False

    for i in range(len(l1) - sublist_length + 1):
        if l1[i:i + sublist_length] == l2:
            return True
    return False


def get_graph_actual_soh(data, frequent_itemset, names, ic, id, vc, vd, tc, td, color, n):
    """
    Строит график SOH с обнаружением аномалий на основе частых паттернов

    :param data: данные ячейки (измерения и SOH)
    :param frequent_itemset: список частых паттернов для данной категории аномалий
    :param names: список уникальных комбинаций параметров
    :param ic, id, vc, vd, tc, td: списки уникальных паттернов измерений
    :param color: цвет для отметки аномалий на графике
    :param n: название категории аномалий (например, "low", "midum", "high")
    :return: кортеж (events, detected_cycles) - найденные паттерны и номера циклов аномалий
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    # Используем известные значения SOH вместо предсказанных
    actual_soh = data["SOH"].values

    # Подготавливаем данные для поиска паттернов
    x1 = prepare_data(data)
    chp = data["charge_policy"][0]

    # Строим график известных значений SOH
    plt.plot(range(len(actual_soh)), actual_soh, label="Actual SOH values", color="blue", linewidth=2)

    events = []
    detected_cycles = []

    # Ищем аномалии в известных данных
    for i in range(len(x1)):
        d = transform_one(x1[i], ic, id, vc, vd, tc, td, names, chp)
        j = 0
        found = False
        limit = False

        # Поиск совпадений с частыми паттернами
        while (not found and not limit):
            if (j >= len(frequent_itemset)):
                limit = True
                break
            else:
                if (search(d, frequent_itemset[j])):
                    found = True
                    break
            j = j + 1

        # Если найден опасный паттерн, отмечаем на графике
        if (found and not limit):
            events.append(frequent_itemset[j])
            index = (x1[i].index)[-1]
            ax.scatter(index + 5, actual_soh[index], color=color, marker='*')
            detected_cycles.append(index)
    if ('index' in locals()):
        ax.scatter(index + 5, actual_soh[index], color=color, marker='*',
                   label="events that lead to " + n + " SOH decrease")
    ax.legend()

    # Добавляем вертикальную линию для обозначения начала анализа (если нужно)
    analysis_start = 10  # Начинаем анализ после первых 10 циклов
    plt.axvline(x=analysis_start, color='orange', linestyle='--', alpha=0.7)
    plt.text(analysis_start + 5, np.mean(actual_soh), 'Analysis start', color="orange")

    ax.legend()
    plt.xlabel("Cycle number")
    plt.ylabel("SOH")
    plt.title(f"SOH with {n} anomaly detection")
    plt.grid(True, alpha=0.3)


    return events, detected_cycles



eventsl=get_graph_actual_soh(data[5],frequent_itemsets_low,names1,ic,id,vc,vd,tc,td,"green","low")
eventsm=get_graph_actual_soh(data[5],frequent_itemsets_midum,names2,ic,id,vc,vd,tc,td,"orange","medium")
eventsh=get_graph_actual_soh(data[5],frequent_itemsets_high,names3,ic,id,vc,vd,tc,td,"red","high")


plt.show()