import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle




def derive(data):
    '''
    Вычисляет производные (разности) значений SOH между соседними циклами

    :param data: список значений SOH ячейки размера n
    :return: список производных SOH размера n-1, где каждый элемент - разность между текущим и предыдущим значением
    '''

    d=[]
    for i in range(1,len(data),1):
        j=(data[i]-data[i-1])
        d.append(j)
    return d


def compare(data1, data2, soh1, soh2):
    '''
    Сравнивает производные SOH двух ячеек и обнаруживает циклы с аномальным падением SOH

    :param data1: список производных SOH эталонной ячейки
    :param data2: список производных SOH проверяемой ячейки
    :param soh1: список SOH эталонной ячейки
    :param soh2: список SOH проверяемой ячейки
    :return: кортеж вида:
    (i + 1, data1[i], data2[i], soh1[i + 1], soh2[i + 1], x)
    где:
        i + 1: номер цикла, где обнаружена аномалия
        data1[i]: производная эталонной ячейки на интервале [i, i+1]
        data2[i]: производная проверяемой ячейки на интервале [i, i+1]
        soh1[i + 1]: SOH эталонной ячейки на цикле i+1
        soh2[i + 1]: SOH проверяемой ячейки на цикле i+1
        x: процент падения SOH
    '''

    size = min(len(data1), len(data2))
    t = abs(np.array(data1[0:size]) - np.array(data2[0:size]))
    p = []
    for i in range(len(t)):
        if t[i] >= 0.001:
            x = 100 - ((soh2[i + 1] * 100) / soh1[i + 1])
            p.append((i + 1, data1[i], data2[i], soh1[i + 1], soh2[i + 1], x))
    return p


def select(d):

    '''
    Фильтрует аномалии, оставляя только те, где падение SOH составляет 1% или более

    :param d: результат работы функции compare()
    :return: отфильтрованный список аномалий с падением SOH >= 1%
    '''

    p=[]
    for i in range(len(d)):
        if d[i][5]>=1:
            p.append(d[i])
    return p


def events_extract(k, data):

    '''
    Извлекает данные за 10 циклов до момента обнаружения аномалии

    :param k: отфильтрованные аномалии (результат select())
    :param data: полные данные проверяемой ячейки
    :return: список вида:
    o = [
        # Событие 1
        данные_циклов_1,      # 10 циклов данных (numpy array)
        (цикл_падения_1, процент_падения_1),  # Метаданные

        # Событие 2
        данные_циклов_2,      # 10 циклов данных (numpy array)
        (цикл_падения_2, процент_падения_2),  # Метаданные

        # Событие 3
        данные_циклов_3,
        (цикл_падения_3, процент_падения_3),
        # ...
    ]
    данные_циклов имеют вид:
    d = np.array([
        [Ic_1, Id_1, Vc_1, Vd_1, Tc_1, Td_1, charge_policy_1],  # Цикл N-10
        [Ic_2, Id_2, Vc_2, Vd_2, Tc_2, Td_2, charge_policy_2],  # Цикл N-9
        ...
        [Ic_10, Id_10, Vc_10, Vd_10, Tc_10, Td_10, charge_policy_10]  # Цикл N-1
    ])
    где
    Ic_i, Id_i, Vc_i, Vd_i, Tc_i, Td_i - списки измеренных данных на (N-11+i)-ом цикле
    charge_policy_i - политика заряда на i-ом цикле
    '''

    o = []
    for j in range(len(k)):
        d = np.array(data[["Ic", "Id", "Vc", "Vd", "Tc", "Td", "charge_policy"]][(k[j][0]-10):k[j][0]])
        if(len(d) != 0):
            o.append(d)
            o.append((k[j][0], k[j][5]))
    return o


def getEvents(data1, data2):

    '''
    Выполняет функции derive, compare, select, events_extract
    для эталонной ячейки и одной из проверяемых ячеек

    :param data1: данные эталонной ячейки
    :param data2: данные проверяемой ячейки
    :return: результат events_extract() - список с данными циклов и метаданными аномалий
    '''

    dtrue_values = derive(data1["SOH"])
    dfalse_values = derive(data2["SOH"])
    p = compare(dtrue_values, dfalse_values, data1["SOH"], data2["SOH"])
    k = select(p)
    o = events_extract(k, data2)
    return o


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

    return (d1[0]==d2[0] and d1[1]==d2[1] and  d1[2]==d2[2] and d1[3]==d2[3] and d1[4]==d2[4] and d1[5]==d2[5])


def createdf(data1,data2):
    '''
    Создает структурированные данные об аномалиях с привязкой к циклам

    :param data1: данные эталонной ячейки
    :param data2: данные проверяемой ячейки
    :return: кортеж (datat, result):
        - datat: данные циклов с номерами циклов
        datat: [
            [  # Событие 1 на цикле N1
                (данные_цикла_N1-10, N1-10),
                (данные_цикла_N1-9, N1-9),
                ...
                (данные_цикла_N1-1, N1-1)
            ],
            [  # Событие 2 на цикле N2
                (данные_цикла_N2-10, N2-10),
                (данные_цикла_N2-9, N2-9),
                ...
                (данные_цикла_N2-1, N2-1)
            ],
            ...
        ]
        - result: список процентов падения SOH для каждого события
    '''

    events=getEvents(data1,data2)
    s=0
    datat=[]
    result=[]
    for i in range(0,len(events),2):
        data=[]
        s=(events[i+1][0])-10 # начальный цикл - за 10 циклов до падения
                              # (events[i+1][0]) - номер цикла падения
        for j in range(len(events[i])):
            data.append((events[i][j],s)) # (данные цикла, номер цикла)
            s=s+1
        result.append(events[i+1][1])
        datat.append(data)
    return datat, result


def getd(df_midum, ic, id, vc, vd, tc, td):
    '''
    Обрабатывает данные циклов, присваивая уникальные идентификаторы схожим графикам

    :param df_midum: DataFrame с данными циклов
    :param ic, id, vc, vd, tc, td: списки для хранения уникальных графиков каждого типа измерений
    :return: обработанные данные с присвоенными идентификаторами графиков
    Total_data_m = [
        Событие 1
        [
            Цикл 1:
            ("ic2", "id5", "vc1", "vd3", "tc0", "td2", 40),

            Цикл 2:
            ("ic2", "id5", "vc1", "vd3", "tc1", "td2", 41),

            Цикл 3:
            ("ic3", "id5", "vc1", "vd4", "tc1", "td2", 42),

            ...  Всего 10 циклов
        ],

        Событие 2
        [
            ("ic0", "id3", "vc2", "vd1", "tc0", "td0", 75),
            ("ic0", "id3", "vc2", "vd1", "tc0", "td0", 76),
            ... Всего 10 циклов
        ]
        ... другие события ...
    ]
    '''

    # df_midum["data"][i][j] = (array([Ic, Id, Vc, Vd, Tc, Td, charge_policy]), номер_цикла)
    Total_data_m = []
    for i in range(len(df_midum)):
        data = []
        for j in range(10):
            find = False
            for k in range(len(ic)):
                if (compare_graphs(ic[k][0], df_midum["data"][i][j][0][0]) and ic[k][1] == df_midum["data"][i][j][0][
                    6]):
                    IC = "ic" + str(k)
                    find = True
                    break

            if (not find):
                ic.append((df_midum["data"][i][j][0][0], df_midum["data"][i][j][0][6]))
                IC = "ic" + str(len(ic) + 1)
            find = False
            for k in range(len(id)):
                if (compare_graphs(id[k][0], df_midum["data"][i][j][0][1]) and id[k][1] == df_midum["data"][i][j][0][
                    6]):
                    ID = "id" + str(k)
                    find = True
                    break
            if (not find):
                id.append((df_midum["data"][i][j][0][1], df_midum["data"][i][j][0][6]))
                ID = "id" + str(len(id) + 1)
            find = False
            for k in range(len(vc)):
                if (compare_graphs(vc[k][0], df_midum["data"][i][j][0][2]) and vc[k][1] == df_midum["data"][i][j][0][
                    6]):
                    VC = "vc" + str(k)
                    find = True
                    break
            if (not find):
                vc.append((df_midum["data"][i][j][0][2], df_midum["data"][i][j][0][6]))
                VC = "vc" + str(len(vc) + 1)
            find = False
            for k in range(len(vd)):
                if (compare_graphs(vd[k][0], df_midum["data"][i][j][0][3]) and vd[k][1] == df_midum["data"][i][j][0][
                    6]):
                    VD = "vd" + str(k)
                    find = True
                    break
            if (not find):
                vd.append((df_midum["data"][i][j][0][3], df_midum["data"][i][j][0][6]))
                VD = "vd" + str(len(vd) + 1)
            find = False
            for k in range(len(tc)):
                if (compare_graphs(tc[k][0], df_midum["data"][i][j][0][4]) and tc[k][1] == df_midum["data"][i][j][0][
                    6]):
                    TC = "tc" + str(k)
                    find = True
                    break
            if (not find):
                tc.append((df_midum["data"][i][j][0][4], df_midum["data"][i][j][0][6]))
                TC = "tc" + str(len(tc) + 1)
            find = False
            for k in range(len(td)):
                if (compare_graphs(td[k][0], df_midum["data"][i][j][0][5]) and td[k][1] == df_midum["data"][i][j][0][
                    6]):
                    TD = "td" + str(k)
                    find = True
                    break
            if (not find):
                td.append((df_midum["data"][i][j][0][5], df_midum["data"][i][j][0][6]))
                TD = "td" + str(len(td) + 1)
            data.append((IC, ID, VC, VD, TC, TD, df_midum["data"][i][j][1]))
        Total_data_m.append(data)
    return Total_data_m


def getting_db(data1, data2, ic, id, vc, vd, tc, td):
    '''
    Обрабатывает пару ячеек и распределяет аномалии по категориям тяжести

    :param data1: эталонная ячейка
    :param data2: проверяемая ячейка
    :param ic, id, vc, vd, tc, td: списки уникальных графиков
    :return: кортеж с данными аномалий трех категорий тяжести,
             где каждый элемент - результат getd()
    '''

    datat, result = createdf(data1, data2)
    df = pd.DataFrame({"data": datat, "result": result})
    soh_bins = [1, 9, 14, np.inf]
    # discretize the soh column
    df['result'] = pd.cut(df['result'], bins=soh_bins, labels=['low', 'midum', 'high'])
    df_low = df[df["result"] == "low"]
    df_midum = df[df["result"] == "midum"]
    df_high = df[df["result"] == "high"]
    df_low = df_low.reset_index(drop=True)
    df_midum = df_midum.reset_index(drop=True)
    df_high = df_high.reset_index(drop=True)

    # df_low - датафрейм вида:
    #    data                   result
    # 0  данные_циклов_1        'low'
    # 1  данные_циклов_2        'low'
    # 2  ...                    ...

    # df_midum - датафрейм вида:
    #    data                   result
    # 0  данные_циклов_1        'midum'
    # 1  данные_циклов_2        'midum'
    # 2  ...                    ...

    # df_high - датафрейм вида:
    #    data                   result
    # 0  данные_циклов_1        'high'
    # 1  данные_циклов_2        'high'
    # 2  ...                    ...


    datal = getd(df_low, ic, id, vc, vd, tc, td)

    datam = getd(df_midum, ic, id, vc, vd, tc, td)

    datah = getd(df_high, ic, id, vc, vd, tc, td)

    return datal, datam, datah


def getS(dataT, dataF):
    '''
    Обрабатывает множество проверяемых ячеек относительно одной эталонной

    :param dataT: эталонная ячейка
    :param dataF: список проверяемых ячеек
    :return: кортеж с данными аномалий всех категорий и списками уникальных графиков,
             где данные аномалий - результат getting_db
    '''

    datam = pd.DataFrame()
    datal = pd.DataFrame()
    datah = pd.DataFrame()
    ic = []
    id = []
    vc = []
    vd = []
    tc = []
    td = []
    for i in range(len(dataF)):
        l, m, h = getting_db(dataT, dataF[i], ic, id, vc, vd, tc, td)
        if (len(l) != 0):
            datal=pd.concat([datal,pd.DataFrame(l)],ignore_index=True)
        if (len(m) != 0):
            datam=pd.concat([datam,pd.DataFrame(m)],ignore_index=True)
        if (len(h) != 0):
            datah=pd.concat([datah,pd.DataFrame(h)],ignore_index=True)

    return datal, datam, datah, ic, id, vc, vd, tc, td


def transfer(df_final):
    '''
    Преобразует данные в формат с уникальными идентификаторами событий

    :param df_final: исходные данные циклов
    :return: кортеж (all_data, names):
        - all_data: преобразованные данные с идентификаторами 'Е'
        - names: список уникальных событий/графиков
    '''

    names=[df_final[0][0]]
    all_data=[]
    for i in range(len(df_final)):
        data=[]
        for j in range(10):
            find=False
            for k in range(len(names)):
                if(equal(names[k],df_final[j][i])):
                    find=True
                    data.append(("E"+str(k),df_final[j][i][6]))
                    break
            if (not find):
                data.append(("E"+str(k+1),df_final[j][i][6]))
                names.append(df_final[j][i])
        all_data.append(data)
    return all_data,names


data = np.load('celles.npy', allow_pickle=True)



dataf = []
for i in range(len(data)):
    if i != 117:
        dataf.append(data[i])
df_final1, df_final2, df_final3, ic, id, vc, vd, tc, td = getS(data[117], dataf)


# Сохраняем каждый список в отдельный файл
with open('ic.pkl', 'wb') as f:
    pickle.dump(ic, f)
with open('id.pkl', 'wb') as f:
    pickle.dump(id, f)
with open('vc.pkl', 'wb') as f:
    pickle.dump(vc, f)
with open('vd.pkl', 'wb') as f:
    pickle.dump(vd, f)
with open('tc.pkl', 'wb') as f:
    pickle.dump(tc, f)
with open('td.pkl', 'wb') as f:
    pickle.dump(td, f)
print("Все файлы успешно сохранены")



all_data1, names1 = transfer(df_final1)
np.save("all_data1.npy", all_data1, allow_pickle=True)
np.save("names1", names1, allow_pickle=True)
print('all_data1, names1 сохранены!')

all_data2, names2 = transfer(df_final2)
np.save("all_data2.npy", all_data2, allow_pickle=True)
np.save("names2", names2, allow_pickle=True)
print('all_data2, names2 сохранены!')

all_data3, names3 = transfer(df_final3)
np.save("all_data3.npy", all_data3, allow_pickle=True)
np.save("names3", names3, allow_pickle=True)
print('all_data3, names3 сохранены!')