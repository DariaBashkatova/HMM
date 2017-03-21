import pandas as pd
from intervals import FloatInterval

# масштабирование от 0 до 100
def Scaling(X, min = 0.0, max = 100.0):
    X_scaled = (max - min) / (X.max(axis=0) - X.min(axis=0)) * (X - X.min(axis=0)) + min
    return X_scaled


# квантование
def Quantization(X):
    # 10 интервалов длиной 10
    intervals = list(map(lambda x: FloatInterval([x, x + 10.0]), range(0, 100, 10)))
    # по строкам в dataframe
    for row in X.itertuples():
        # по элементам в строке; по индексу 0 - номер строки в dataframe, далее элементы X1, X2, ..., X100
        for e in range(1, len(X.count(axis=0)) + 1):
            # для каждого интервала
            for inter in intervals:
                # если элемент принадлежит интервалу, заменяем этот элемент в dataframe на значение середины этого интервала
                if row[e] in inter:
                    X.iloc[row[0], e - 1] = inter.centre
                    break
    return X

# считывание данных
df_train = pd.read_csv('Training-data.csv')
df_test = pd.read_csv('Testing-data.csv')


# разбиение выборки на признаки и ответы
X_train, y_train = df_train.loc[:, 'X1':'X100'], df_train.loc[:, 'class':]
X_test, y_test = df_test.loc[:, 'X1':'X100'], df_test.loc[:, 'class':]

# масштабирование и квантование
X_train_scale = Quantization(Scaling(X_train))
X_test_scale = Quantization(Scaling(X_test))

df_train = pd.concat([X_train_scale, y_train], axis=1).sort_values(by='class')

X_train_cavity = df_train.loc[df_train['class'] == 0, 'X1':'X100']
X_train_hill = df_train.loc[df_train['class'] == 1, 'X1':'X100']

# X_hill_train, y_hill_train = df_train.loc[df_train['class'] == 1, 'X1':'X100'], df_train.loc[df_train['class'] == 1, 'class':]
# print(X_cavity_train, y_cavity_train)

#
#
# # разбиение тестовой выборки на признаки и ответы
# X_cavity_test, y_cavity_test = df_test.loc[df_test['class'] == 0, 'X1':'X100'], df_test.loc[df_test['class'] == 0, 'class':]
# X_hill_test, y_hill_test = df_test.loc[df_test['class'] == 1, 'X1':'X100'], df_test.loc[df_test['class'] == 1, 'class':]
#

    # , X_hill_train_scale= Quantization(Scaling(X_hill_train))
