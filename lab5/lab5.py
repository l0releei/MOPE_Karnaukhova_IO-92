import random as r
import numpy as np
import pprint
from scipy.stats import t, f
import sklearn.linear_model as lm
from functools import partial
from prettytable import PrettyTable
import time

table0 = PrettyTable()
table0.field_names = (["Студент", "Группа"])
name = "Карнаухова Анастасія"
group = "ІО-92"
table0.add_row([name, group])
print(table0)


x_range = [[-6, 10], [-3, 5], [-4, 9]]
x_sered_max = sum([x[1] for x in x_range])/3
x_sered_min = sum([x[0] for x in x_range])/3

x01 = (x_range[0][1] - x_range[0][0]) / 2
x02 = (x_range[1][1] - x_range[1][0]) / 2
x03 = (x_range[2][1] - x_range[2][0]) / 2
delta_x1 = x_range[0][1] - x01
delta_x2 = x_range[1][1] - x02
delta_x3 = x_range[2][1] - x03

y_max = 200 + x_sered_max
y_min = 200 + x_sered_min

def create_plan_matrix(n, m):
    y = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            y[i][j] = r.randint(int(y_min), int(y_max))
    x_matrix_norm = [
        [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1],
        [1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
        [1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
        [1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0, 0],
        [1, 1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0, 0],
        [1, 0, -1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0],
        [1, 0, 1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0],
        [1, 0, 0, -1.215, 0, 0, 0, 0, 0, 0, 1.4623],
        [1, 0, 0, 1.215, 0, 0, 0, 0, 0, 0, 1.4623],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    print('Нормована матриця:')
    pprint.pprint(x_matrix_norm)

    x_matrix = [[] for x in range(n)]
    for i in range(len(x_matrix)):
        if i < 8:
            x1 = x_range[0][0] if x_matrix_norm[i][1] == -1 else x_range[0][1]
            x2 = x_range[1][0] if x_matrix_norm[i][2] == -1 else x_range[1][1]
            x3 = x_range[2][0] if x_matrix_norm[i][3] == -1 else x_range[2][1]
        else:
            x1 = x_matrix_norm[i][1] * delta_x1 + x01
            x2 = x_matrix_norm[i][2] * delta_x2 + x02
            x3 = x_matrix_norm[i][3] * delta_x3 + x03
        x_matrix[i] = [1, float(format(x1, '.2f')),
                       float(format(x2, '.2f')),
                       float(format(x3, '.2f')),
                       float(format(x1 * x2, '.2f')),
                       float(format(x1 * x3, '.2f')),
                       float(format(x2 * x3, '.2f')),
                       float(format(x1 * x2 * x3, '.2f')),
                       float(format(x1 ** 2, '.2f')),
                       float(format(x2 ** 2, '.2f')),
                       float(format(x3 ** 2, '.2f'))]
    print('Натуралізована матриця: ')
    pprint.pprint(x_matrix)

    print('Y :')
    pprint.pprint(y)

    y_avr = np.zeros(n)
    for i in range(len(y)):
        for j in range(len(y[0])):
            y_avr[i] += y[i][j]/ m
    return [x_matrix_norm, x_matrix, y, y_avr]

def find_coefs(x, y):
    skm = lm.LinearRegression(fit_intercept=False)  # знаходимо коефіцієнти рівняння регресії
    skm.fit(x, y)
    B = skm.coef_
    print('Коефіціенти: ')
    print(B)
    t = time.time() - start_time
    print("Час пошуку: ", t)
    if t > 0.1:
        B = False
    return B

def perevirka(x, y, b):
    y_pract = np.zeros(len(y))
    for i in range(len(x)):
        for j in range(len(x[0])):
            y_pract[i] += b[j] * x[i][j]
    print("\nПеревірка:")
    print("\ny - real :" , y)
    print('\ny - found:', y_pract)


def get_new_y(x, b):
    y_pract = np.zeros(len(y))
    for i in range(len(x)):
        for j in range(len(x[0])):
            y_pract[i] += b[j] * x[i][j]
    return y_pract

def get_cohren_critical(prob, f1, f2):
    f_crit = f.isf((1 - prob) / f2, f1, (f2 - 1) * f1)
    return f_crit / (f_crit + f2 - 1)

def cohren_crit(y, n, m):
    y_var = [np.var(i) for i in y]
    Gp = max(y_var)/sum(y_var)
    Gt = get_cohren_critical(0.95, m-1, n)
    if(Gp < Gt):
        print("\nДисперсії однорідні")
        return True
    else:
        print("\nДисперсії не однорідні")
        return False

fisher_teor = partial(f.ppf, q=1 - 0.05)

student_teor = partial(t.ppf, q=1 - 0.025)

def kriteriy_studenta(x, y, y_aver, n, m, B):
    d = 0
    y_var = [np.var(i) for i in y]
    s_kv_aver = sum(y_var) / n
    s_aver = (s_kv_aver/(n * m))** 0.5
    b = np.zeros(len(x[0]))
    for i in range(len(x[0])):
        for j in range(len(y_aver)):
            b[i] += y_aver[j] * x[j][i] / n
    ts = []
    for bi in b:
        ts.append(bi/s_aver)
    Stud_teor = student_teor(df = (m-1) * n)
    for i in range(len(ts)):
        if ts[i] < Stud_teor:
            B[i] = 0
        else:
            d += 1
    print("\nКоефіціенти після перевірки нуль гіпотези: ")
    print(B)
    return [B, d]

def kriteriy_fishera(m, n, d, new_y_pract, y_avr, y):
    f4 = n - d
    f3 = (m-1)*n
    y_var = [np.var(i) for i in y]
    Sa = (sum(y_var)/ n)
    Sad = m/(n-d)* sum([(new_y_pract[i] - y_avr[i])**2 for i in range(len(y_avr))])
    pract = Sad/Sa
    teor = fisher_teor(dfn= f4, dfd=f3)
    if pract > teor:
        print("\nПрактичне значення:" , pract)
        print("Теоретичне значення:" , teor)
        print("\nРівняння регресії неадекватне")
        return [False, False]
    else:
        print("\nРівняння регресії адекватне")
        return [True, True]




if __name__ == "__main__":
    odnorid = False
    adekvat = False
    n = 15
    m = 3

    while not adekvat:

        while not odnorid:
            x_matrix_norm, x_matrix, y, y_avr = create_plan_matrix(n, m)

            odnorid = cohren_crit(y, n, m)

            if odnorid == False:
                m+=1
        B = find_coefs(x_matrix,    y_avr)
                if B.all() != False or type(B) == list:

            perevirka(x_matrix, y_avr, B)
            new_B, d = kriteriy_studenta(x_matrix_norm, y, y_avr, n, m, B)
            new_y_pract = get_new_y(x_matrix, new_B)

            adekvat, odnorid = kriteriy_fishera(m, n, 4, new_y_pract, y_avr, y)

        else:
            print("Затраченний час більше за 0.1, тому модель неадекватна")
            adekvat, odnorid = False, False
