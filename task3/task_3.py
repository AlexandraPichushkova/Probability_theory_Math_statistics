import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import scipy
import math
import os

TABLE = True           # генерировать ли таблицу
FILENAME = 'cat.txt'   # имя таблицы
COUNT = 100            # число повторов
PRINT = False          # печатать ли данные
HIST = False           # строить ли гистограмму
SUPHIST = False
REGR = False           # строить ли регресии

# Константы для задач
R1, G1, B1 = 11, 10, 11
R2, G2, B2 = 10, 9, 10
R3, G3, B3 = 9, 11, 5

P001 = 0.99
P0001 = 0.999

def save_plot(name='', fmt='png'):
    # return
    pwd = os.getcwd()
    iPath = './{}'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), dpi=150, bbox_inches='tight')
    os.chdir(pwd)


def generate_next_customer_time(time):
    return time + random.randint(0, R1 + G1 + B1)


def generate_processing_time__1(time):
    return time + random.randint(R1 + G1, R1 + G1 + 2 * B1)


def generate_processing_time__2(time):
    return time + random.randint(R1 + G1, 2 * R1 + G1 + B1)


def addCustomer():
    global seque1, seque2
    if seque1 <= seque2:
        seque1 += 1
    else:
        seque2 += 1


def averageSum(v, n):
    return sum(v[:n]) / n


def dispersion(v, n):
    avg = averageSum(v, n)
    q = 0

    for i in range(n): q += (v[i] - avg) ** 2
    return q / (n - 1)


def avgSquareDev(v, n):
    return dispersion(v, n) ** 0.5


def covariation(v1, v2, n):
    avg1 = averageSum(v1, n)
    avg2 = averageSum(v2, n)
    q = 0

    for i in range(n): q += (v1[i] - avg1) * (v2[i] - avg2)
    return q / (n - 1)


def normalCorellation(v1, v2, n):
    return covariation(v1, v2, n) / avgSquareDev(v1, n) / avgSquareDev(v2, n)


def calculate_t_value(P, n):
    return scipy.stats.t.ppf((1 + P) / 2, n - 1)


def calc_confidence_interval(v, n, P):
    avg = averageSum(v, n)
    div = avgSquareDev(v, n)
    ti = calculate_t_value(P, n) * div / n ** 0.5

    return round(avg - ti, 3), round(avg + ti, 3)


def regression(v1, v2, n, name1, name2):
    cor = normalCorellation(v1, v2, n)
    div1 = avgSquareDev(v1, n)
    div2 = avgSquareDev(v2, n)
    avg1 = averageSum(v1, n)
    avg2 = averageSum(v2, n)

    a = cor * div2 / div1
    b = avg2 - a * avg1

    q = min(v1)
    mv1 = max(v1)
    e = (max(v1) / min(v1)) / 200
    x = []
    y = []
    while q <= mv1:
        x.append(q)
        y.append(a * q + b)
        q += e

    plt.figure(figsize=(16, 8))
    plt.plot(v1, v2, ' o', alpha=0.7, label=name1 + '/' + name2, lw=1.5, mec='b', mew=2, ms=10)
    plt.plot(x, y, '-', alpha=0.7, label='y = ' + str(round(a, 3)) + 'x + ' + str(round(b, 3)), lw=2, mec='b', mew=2,
             ms=10)

    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(name1 + '/' + name2)
    plt.grid()
    plt.legend()

    save_plot(name=name1 + '_' + name2, fmt='png')
    plt.show()


def find_index(y, value):
    for i in range(1, len(y)):
        if value < y[i]: return i - 1
    return len(y) - 2


def z(x, avg, div):
    return (x - avg) / div


def calc_prob_density_normal(z):
    return math.e ** (-(z ** 2 / 2)) / (2 * math.pi) ** 0.5


def n_i(n, x, avg, div, h):
    return h * 100 / div * calc_prob_density_normal(z(x, avg, div))


def plot_histogram(v, name, start, step, count, minValue):
    y = [start]
    q = start

    for i in range(count):
        q += step
        y.append(q)
    x = [0 for i in range(len(y) - 1)]

    for i in v: x[find_index(y, i)] += 1

    while x[-1] < 5:
        x[-2] += x[-1]
        x.pop()
        y.pop()

    while x[0] < 5:
        x[1] += x[0]
        x.pop(0)
        y.pop(0)

    cntry = []
    for i in range(1, len(y)):
        cntry.append(y[i - 1] + (y[i] - y[i - 1]) / 2)

    n = []
    for i in range(len(x)):
        n.append(n_i(x[i], cntry[i], averageSum(v, len(v)), avgSquareDev(v, len(v)), step))
    k = len(x) - 2 - 1

    X = 0
    for i in range(len(x)):
        X += (x[i] - n[i]) ** 2 / n[i]
    P = [3.8, 6.0, 7.8, 9.5, 11.1, 12.6, 14.1, 15.5, 16.9,
         18.3, 19.7, 21.0, 22.4, 23.7, 25.0, 26.3, 27.6, 27.6, 28.9, 30.1,
         31.4, 32.7, 33.9, 35.2, 36.4, 37.7]

    if SUPHIST:
        plt.figure(figsize=(16, 8))
        plt.hist(v)
    plt.figure(figsize=(16, 8))
    plt.bar(cntry, x, edgecolor='white', width=step, label='Xcalc = ' + str(round(X, 2)) + '\nXctrl = ' + str(P[k - 1]))

    plt.xlabel(name)
    plt.ylabel('Count')
    plt.title(name)
    plt.legend()

    save_plot(name=name, fmt='png')
    plt.show()

    return x, y


def calc_laplace_value(x, M, S):
    x = (x - M) / S
    # print(x)
    phi = (scipy.stats.norm.cdf(x) - 0.5)

    return phi


def plot_histogram_with_laplace(v, name, start, step, count, minValue):
    y = [start]
    q = start

    for i in range(count):
        q += step
        y.append(q)
    x = [0 for i in range(len(y) - 1)]

    # print(y)
    # print(x)

    y = [0.965, 0.969, 0.974, 0.978, 0.986, 0.99, 0.994, 1.0, 1.005,
         1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045]

    for i in v: x[find_index(y, i)] += 1

    while x[-1] < 5:
        x[-2] += x[-1]
        x.pop()
        y.pop()

    while x[0] < 5:
        x[1] += x[0]
        x.pop(0)
        y.pop(0)

    w = []
    for i in range(1, len(y)):
        w.append(y[i] - y[i - 1])

    cntry = []
    for i in range(1, len(y)):
        cntry.append(y[i - 1] + (y[i] - y[i - 1]) / 2)

    n = []
    for i in range(len(x)):
        n.append(n_i(x[i], cntry[i], averageSum(v, len(v)), avgSquareDev(v, len(v)), w[i]))
    print()
    print()
    P = []
    y2 = [0] + y + [10 ** 5]
    for i in range(1, len(y2)):
        P.append(calc_laplace_value((y2[i]), averageSum(v, len(v)), avgSquareDev(v, len(v))) - calc_laplace_value((y2[i - 1]), averageSum(v, len(v)),
                                                                                    avgSquareDev(v, len(v))))
    # print(n)
    print(P)
    print()
    k = len(x) - 2 - 1
    # print(n)

    X = 0
    for i in range(len(x)):
        X += (x[i] - n[i]) ** 2 / n[i]
    # print('X =', X)
    # красный
    P = [3.8, 6.0, 7.8, 9.5, 11.1, 12.6, 14.1, 15.5, 16.9,
         18.3, 19.7, 21.0, 22.4, 23.7, 25.0, 26.3, 27.6, 27.6, 28.9, 30.1,
         31.4, 32.7, 33.9, 35.2, 36.4, 37.7]

    # print(x, w)
    # print()
    # print(len(x), len(w))

    if SUPHIST:
        plt.figure(figsize=(16, 8))
        plt.hist(v)
    plt.figure(figsize=(16, 8))
    plt.bar(cntry, x, edgecolor='white', width=w, label='Xcalc = ' + str(round(X, 2)) + '\nXctrl = ' + str(P[k - 1]))

    plt.xlabel(name)
    plt.ylabel('Count')
    plt.title(name)
    plt.legend()
    # plt.text(str(X)+' '+str(P[k-1]))

    save_plot(name=name, fmt='png')
    plt.show()

    return x, y


def plot_histogram_with_laplace_2(v, name, start, step, count, minValue):
    y = [start]
    q = start

    for i in range(count):
        q += step
        y.append(q)


    y = [0.88, 0.89, 0.9, 0.92, 0.94, 0.97, 0.98, 0.99, 1.0]
    x = [0 for i in range(len(y) - 1)]

    for i in v: x[find_index(y, i)] += 1

    while x[-1] < 5:
        x[-2] += x[-1]
        x.pop()
        y.pop()

    while x[0] < 5:
        x[1] += x[0]
        x.pop(0)
        y.pop(0)

    w = []
    for i in range(1, len(y)):
        w.append(y[i] - y[i - 1])

    cntry = []
    for i in range(1, len(y)):
        cntry.append(y[i - 1] + (y[i] - y[i - 1]) / 2)
    n = []
    for i in range(len(x)):
        n.append(n_i(x[i], cntry[i], averageSum(v, len(v)), avgSquareDev(v, len(v)), w[i]))
    k = len(x) - 2 - 1
    # print(n)

    X = 0
    for i in range(len(x)):
        X += (x[i] - n[i]) ** 2 / n[i]

    P = [3.8, 6.0, 7.8, 9.5, 11.1, 12.6, 14.1, 15.5, 16.9,
         18.3, 19.7, 21.0, 22.4, 23.7, 25.0, 26.3, 27.6, 27.6, 28.9, 30.1,
         31.4, 32.7, 33.9, 35.2, 36.4, 37.7]

    print(x, w)
    print()
    print(len(x), len(w))

    if SUPHIST:
        plt.figure(figsize=(16, 8))
        plt.hist(v)
    plt.figure(figsize=(16, 8))
    plt.bar(cntry, x, edgecolor='white', width=w, label='Xcalc = ' + str(round(X, 2)) + '\nXctrl = ' + str(P[k - 1]))

    plt.xlabel(name)
    plt.ylabel('Count')
    plt.title(name)
    plt.legend()

    save_plot(name=name, fmt='png')
    plt.show()

    return x, y


SAL1 = []
SAL2 = []
SK1 = []
SK2 = []

for i in range(COUNT):
    # изменяем семя
    random.seed(i)

    seque1 = 0  # первая очередь
    seque2 = 0  # вторая очередь
    processingTime1 = 0  # момент, когда обработается ещё одна заявка
    processingTime2 = 0  # момент, когда обработается ещё одна заявка
    eventTime = generate_next_customer_time(1)  # время следующего посетителя

    time = 1
    endTime = 1 * 60 * 60

    # для подсчёта средней длинны
    sequeTotalLenght1 = 0
    sequeTotalLenght2 = 0
    # для подсчёта коэффициента загрузки
    util1 = 0
    util2 = 0

    while time <= endTime:

        if processingTime1 >= time: util1 += 1
        if processingTime2 >= time: util2 += 1

        # если одна из касс выполнила заказ
        if processingTime1 == time: seque1 = max(seque1 - 1, 0)
        if processingTime2 == time: seque2 = max(seque2 - 1, 0)

        # если пришёл ещё один покупатель
        while eventTime == time:
            addCustomer()
            eventTime = generate_next_customer_time(time)

        # если касса должна что-то делать, то она начинает (быстро-быстро)
        if seque1 != 0 and processingTime1 <= time: processingTime1 = generate_processing_time__1(time)
        if seque2 != 0 and processingTime2 <= time: processingTime2 = generate_processing_time__2(time)

        sequeTotalLenght1 += seque1
        sequeTotalLenght2 += seque2

        # время назад не вернуть
        time += 1

    # считает то, что хотели посчитать
    SequeAverageLenght1 = sequeTotalLenght1 / endTime
    SequeAverageLenght2 = sequeTotalLenght2 / endTime
    SequeK1 = util1 / endTime
    SequeK2 = util2 / endTime

    # добовляем всё в списки
    SAL1.append(SequeAverageLenght1)
    SAL2.append(SequeAverageLenght2)
    SK1.append(SequeK1)
    SK2.append(SequeK2)

    if PRINT:
        print('SequeAverageLenght1 =', SequeAverageLenght1)
        print('SequeAverageLenght2 =', SequeAverageLenght2)
        print()

        print('SequeK1 =', SequeK1)
        print('SequeK2 =', SequeK2)
        print()

if HIST:
    # hist(SAL1, 'AVGC1', 0, 0.8, 16, 5)
    # hist(SAL1, 'AVGC1', 32, 2, 16, 3)
    # hist(SAL2, 'AVGC2', 0, 0.8, 16, 5)
    plot_histogram_with_laplace(SK1, 'UTIL1', 0.965, 0.005, 16, 5)
    # hist3(SK2, 'UTIL2', 0.88, 0.01, 12, 5)

if TABLE:
    q = []
    for i in range(COUNT):
        q.append(str(SAL1[i]) + '\t' + str(SAL2[i]) + '\t' + str(SK1[i]) + '\t' + str(SK2[i]))
    open(FILENAME, 'w').write('\n'.join(q).replace('.', ','))

    n = [10, 25, 50, 100]

    avg = []
    dis = []
    div = []
    cor = []
    ncor = []
    TI_P001 = []
    TI_P0001 = []

    for i in n:
        avg.append([averageSum(SAL1, i), averageSum(SAL2, i), averageSum(SK1, i), averageSum(SK2, i)])
        dis.append([dispersion(SAL1, i), dispersion(SAL2, i), dispersion(SK1, i), dispersion(SK2, i)])
        div.append([avgSquareDev(SAL1, i), avgSquareDev(SAL2, i), avgSquareDev(SK1, i), avgSquareDev(SK2, i)])
        cor.append(
            [covariation(SAL1, SAL2, i), covariation(SK1, SK2, i), covariation(SAL1, SK1, i), covariation(SAL2, SK2, i),
             covariation(SAL1, SK2, i), covariation(SAL2, SK1, i)])
        ncor.append([normalCorellation(SAL1, SAL2, i), normalCorellation(SK1, SK2, i), normalCorellation(SAL1, SK1, i),
                     normalCorellation(SAL2, SK2, i), normalCorellation(SAL1, SK2, i), normalCorellation(SAL2, SK1, i)])

        TI_P001.append([calc_confidence_interval(SAL1, i, P001), calc_confidence_interval(SAL2, i, P001), calc_confidence_interval(SK1, i, P001),
                        calc_confidence_interval(SK2, i, P001)])
        TI_P0001.append([calc_confidence_interval(SAL1, i, P0001), calc_confidence_interval(SAL2, i, P0001), calc_confidence_interval(SK1, i, P0001),
                         calc_confidence_interval(SK2, i, P0001)])

    open('AVG_SUM_' + FILENAME, 'w').write(
        '\n'.join(list(map(lambda x: ' '.join(list(map(lambda y: str(y), x))), avg))).replace('.', ','))
    open('DISPERSION_' + FILENAME, 'w').write(
        '\n'.join(list(map(lambda x: ' '.join(list(map(lambda y: str(y), x))), dis))).replace('.', ','))
    open('DIVIATION_' + FILENAME, 'w').write(
        '\n'.join(list(map(lambda x: ' '.join(list(map(lambda y: str(y), x))), div))).replace('.', ','))
    open('COVARIATION_' + FILENAME, 'w').write(
        '\n'.join(list(map(lambda x: '\t'.join(list(map(lambda y: str(y), x))), cor))).replace('.', ','))
    open('NORMAL_CORELLATION_' + FILENAME, 'w').write(
        '\n'.join(list(map(lambda x: '\t'.join(list(map(lambda y: str(y), x))), ncor))).replace('.', ','))
    open('TI_P001_' + FILENAME, 'w').write(
        '\n'.join(list(map(lambda x: '\t'.join(list(map(lambda y: str(y), x))), TI_P001))).replace('.', ','))
    open('TI_P0001_' + FILENAME, 'w').write(
        '\n'.join(list(map(lambda x: '\t'.join(list(map(lambda y: str(y), x))), TI_P0001))).replace('.', ','))

if REGR:
    regression(SAL1, SAL2, 100, 'AVGC1', 'AVGC2')
    regression(SK1, SK2, 100, 'UTIL1', 'UTIL2')
    regression(SAL1, SK1, 100, 'AVGC1', 'UTIL1')
    regression(SAL2, SK2, 100, 'AVGC2', 'UTIL2')
    regression(SAL1, SK2, 100, 'AVGC1', 'UTIL2')
    regression(SAL2, SK1, 100, 'AVGC2', 'UTIL1')
