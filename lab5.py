import pandas as pd
from intervals import FloatInterval
import numpy as np
import matplotlib.pyplot as plt

# масштабирование от 0 до 10
def Scaling(X, min = 0.0, max = 10.0):
    X = list(map(lambda x: (max - min) / (np.ndarray.max(x) - np.ndarray.min(x)) * (x - np.ndarray.min(x)) + min, X))
    for i in X:
        i[i >= 10.0] = 9.0
    return X


# квантование
def Quantization(X, M, max = 10.0):
    # 10 интервалов длиной 10
    intervals = list(map(lambda x: FloatInterval([x, x + max / M]), np.arange(0, max, max / M)))
    print(intervals)
    # по строкам в dataframe
    for row in X.itertuples():
        # по элементам в строке; по индексу 0 - номер строки в dataframe, далее элементы X1, X2, ..., X100
        for e in range(1, len(X.count(axis=0)) + 1):
            # для каждого интервала
            for inter in intervals:
                # если элемент принадлежит интервалу, заменяем этот элемент в dataframe на значение середины этого интервала
                if row[e] in inter:
                    X.iloc[row[0], e - 1] = inter.lower * M / max
                    break
    return X


# инициализация массива альфа
def alpha_init(o, pi, b, N):
    T = len(o)
    alpha = np.zeros((T, N))
    for i in range(N):
        y = b[i][o[0]]
        alpha[0][i] = pi[i] * b[i][o[0]]
    return alpha


# вычисление альфа
def get_alpha(o, pi, a, b, N):
    alphaZero = alpha_init(o, pi, b, N)
    for t in range(len(alphaZero) - 1):
        for i in range(len(alphaZero[t])):
            alphaZero[t + 1][i] = b[i][o[t + 1]] * sum(alphaZero[t][j] * a[j][i] for j in range(N))
    return alphaZero


# инициализация массива бета
def beta_init(T, N):
    beta = np.zeros((T, N))
    for i in range(N):
        beta[T - 1][i] = 1
    return beta


# вычисление бета
def get_beta(o, a, b, N):
    T = len(o)
    beta = beta_init(T, N)
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                beta[T - t - 1][i] += beta[T - t][j] * b[j][o[T - t]] * a[i][j]
    return beta


# вычисление гамма
def get_gamma(alpha, beta, P, T, N):
    gamma = np.zeros((T, N))
    for t in range(T):
        for i in range(N):
            gamma[t][i] = alpha[t][i] * beta[t][i] / P
    return gamma


# вычисление кси
def get_ksi(o, alpha, beta, P, a, b, N):
    T = len(o)
    ksi = np.zeros(((T - 1, N, N)))
    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                ksi[t][i][j] = (alpha[t][i] * a[i][j] * b[j][o[t + 1]] * beta[t + 1][j]) / P
    return ksi


# Оценка параметров матрицы А
def estimate_A(gamma, ksi, T, K, N):
    a_h = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            num, den = 0, 0
            for k in range(K):
                for t in range(T - 1):
                    num += ksi[k][t][i][j]
                    den += gamma[k][t][i]
            a_h[i][j] = num / den
    return a_h


# Оценка параметров матрицы В
def estimate_B(o, gamma, ksi, T, K, N, M):
    b_h = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            num, den = 0, 0
            for k in range(K):
                for t in range(T - 1):
                    if j == o[k][t]:
                        num += gamma[k][t][i]
                    den += gamma[k][t][i]
            b_h[i][j] = num / den
    return b_h


def get_estimane(o, N, M, K, T):
    ln0, lnk, eps = -1, 0, 1e-2
    A = np.full((N, N), 1 / N)
    B = np.full((N, M), 1 / M)
    pi = np.full(N, 1 / N)
    iter = 0
    while (abs(lnk - ln0) > eps):
        gamma = []
        ksi = []
        ln0 = lnk
        lnk = 0
        for k in range(K):
            alpha = get_alpha(o[k], pi, A, B, N)
            beta = get_beta(o[k], A, B, N)
            P = sum(sum(alpha))
            # print(P)
            lnk += np.log(P)
            gamma.append(get_gamma(alpha, beta, P, T, N))
            ksi.append(get_ksi(o[k], alpha, beta, P, A, B, N))
        # print('a', alpha)
        # print('b', beta)
        A = estimate_A(gamma, ksi, T, K, N)
        B = estimate_B(o, gamma, ksi, T, K, N, M)
        pi = [sum(gamma[k][0][i] for k in range(K)) / K for i in range(N)]
        iter += 1
        print(iter)
    return A, B, pi


def graf(y, name):
    plt.title(name)
    x = np.arange(0, len(y[0]), 1)
    plt.plot(x, y[0], marker='o')
    plt.show()


# считывание данных
df_train = pd.read_csv('Training-data.csv')
# df_test = pd.read_csv('Testing-data.csv')

# разбиение выборки на признаки и ответы
X_train, y_train = df_train.loc[:, 'X1':'X100'], df_train.loc[:, 'class':]
# X_test, y_test = df_test.loc[:, 'X1':'X100'], df_test.loc[:, 'class':]

# M - размер алфавита, N - количество скрытых состояний
M, N = 10, 3

# масштабирование и квантование с шагом 10/M
graf(X_train.values, "Obichnie")
X_train_scale = Quantization(pd.DataFrame(Scaling(X_train.values)), M)

graf(X_train.values, "Normirovanie")

# соединение признаков и ответов, чтобы отсортировать выборку по классам (0/1)
df_train = pd.concat([X_train_scale, y_train], axis=1).sort_values(by='class')

# выделение подвыборок "холмы" и "впадины"
X_train_cavity = df_train.loc[df_train['class'] == 0].values[:, 0:100]
X_train_hill = df_train.loc[df_train['class'] == 1].values[:, 0:100]

X_train_cavity = np.array(list(map(lambda x: x[::3], X_train_cavity)))
X_train_hill = np.array(list(map(lambda x: x[::3], X_train_hill)))

# T - длина последовательности, K - количество последовательностей
T, K = len(X_train_cavity[0]), len(X_train_cavity)

graf(X_train_cavity, "cavity")
graf(X_train_hill, "hill")

A_h_cavety, B_h_cavety, pi_h_cavety = get_estimane(X_train_cavity, N, M, K, T)
print(A_h_cavety, B_h_cavety, pi_h_cavety)
for i in range(len(B_h_cavety)):
    print(sum(B_h_cavety[i]))

# max = 10
# intervals = list(map(lambda x: FloatInterval([x, x + max / M]), np.arange(0, max, max / M)))
# for i in range(len(intervals)):
#     print(intervals[i].lower)
# print(10.0 in intervals[9])
