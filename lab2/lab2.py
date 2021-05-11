import random as ran
import math as ma
import numpy
from prettytable import PrettyTable

table0 = PrettyTable()
table0.field_names = (["Студент", "Группа"])
name = "Карнаухова Анастасія"
group = "ІО-92"
table0.add_row([name, group])
print(table0)

def inf(count, mean):
    array_x1 = [10, 60]
    m = 5 + count
    array_x2 = [-35, 15]
    array_y = [(30 - 11) * 10, (20 - 11) * 10]
    teta0 = ma.sqrt(2 * (2 * m - 2) / (m * (m - 4)))
    matr_x = []
    matr_x2 = []
    matr_y = [[ran.randint(array_y[1], array_y[0]) for j in range(m)] for i in range(3)]
    avey = []
    sigma = []
    array_fuv = []
    array_a = []
    array_aij = []
    for i in range(3):

        if i == 0:
            matr_x.append([-1, -1])
            matr_x2.append([min(array_x1), min(array_x2)])
        elif i == 1:
            matr_x.append([-1, 1])
            matr_x2.append([min(array_x1), max(array_x2)])
        elif i == 2:
            matr_x.append([1, -1])
            matr_x2.append([max(array_x1), min(array_x2)])
    for i in range(len(matr_y)):
        sumer = 0
        temp = sum(matr_y[i]) / len(matr_y[i])
        avey.append(temp)
        for j in range(len(matr_y[i])):
            sumer += (matr_y[i][j] - temp) ** 2
        sigma.append(sumer / len(matr_y[i]))
    array_fuv.append(sigma[0] / sigma[1])
    array_fuv.append(sigma[2] / sigma[0])
    array_fuv.append(sigma[2] / sigma[1])
    array_tetas = [(m - 2) / m * array_fuv[i] for i in range(len(array_fuv))]
    array_ruv = [ma.fabs(i - 1) / teta0 for i in array_tetas]
    for i in range(len(array_ruv)):
        if array_ruv[i] < 2:
            mean = True
            print("R{0}uv > 2".format(i))
        else:
            mean = False
            print("R{0}uv < 2".format(i))
    if mean:
        trans = numpy.array(matr_x).transpose()
        array_mx = [sum(trans[i]) / len(trans[i]) for i in range(2)]
        my = sum(avey) / len(avey)
        for i in range(2):
            temp = 0
            if i == 1:
                for j in matr_x:
                    temp += numpy.array(j).prod()
                array_a.append(temp / 3)
                temp = 0
            for j in range(len(trans[i])):
                temp += (trans[i][j] ** 2)
            array_a.append(temp / 3)
        for i in range(2):
            temp = 0
            for j in range(len(trans[i])):
                temp += trans[i][j] * avey[j]
            array_aij.append(temp / 3)
        first = numpy.array(
            [[1, array_mx[0], array_mx[1]], [array_mx[0], array_a[0], array_a[1]],
             [array_mx[1], array_a[1], array_a[2]]])
        second = numpy.array([my, array_aij[0], array_aij[1]])
        res = numpy.linalg.solve(first, second)
        array_delx = [(max(array_x1) - min(array_x1)) / 2, (max(array_x2) - min(array_x2)) / 2]
        array_zerx = [sum(array_x1) / 2, sum(array_x2) / 2]
        a0 = res[0] - res[1] * (array_zerx[0] / array_delx[0]) - res[2] * (array_zerx[1] / array_delx[1])
        a1 = res[1] / array_delx[0]
        a2 = res[2] / array_delx[1]
        ta = PrettyTable()
        ta.field_names = ["X1", "X2", "Y1", "Y2", "Y3", "Y4", "Y5"]
        ta.add_rows(
            [
                [matr_x[0][0], matr_x[0][1], matr_y[0][0], matr_y[0][1], matr_y[0][2], matr_y[0][3], matr_y[0][4]],
                [matr_x[1][0], matr_x[1][1], matr_y[1][0], matr_y[1][1], matr_y[1][2], matr_y[1][3], matr_y[1][4]],
                [matr_x[2][0], matr_x[2][1], matr_y[2][0], matr_y[2][1], matr_y[2][2], matr_y[2][3], matr_y[2][4]],
            ]
        )
        print("m = ", m)
        print("Матриця планування для m = 5")
        print(ta)
        print("Нормованні значення X1 та X2:\n", matr_x)
        print("Значення функції відгуку при m = {0}:\n".format(m), numpy.array(matr_y))
        print("Середнє значення функції відгуку:\n", avey)
        print("Матиматичне очікування X1 та X2:\n", array_mx)
        print("Значення а:\n", array_a)
        print("Значення aij:\n", array_aij, "\n")
        print("Нормоване рівняння регресії")
        print("y = {0} + {1}*x1 + {2}*x2\n".format(res[0], res[1], res[2]))
        print("Значення дисперсій:\n", sigma)
        print("Значення Fuv:\n", array_fuv)
        print("Значення θuv:\n", array_tetas)
        print("Значення Ruv:\n", array_ruv)

        print("Зробимо перевірку:")
        for i in range(len(matr_x)):
            check = res[0] + res[1] * matr_x[i][0] + res[2] * matr_x[i][1]
            print("y{0} = {1}".format(i, check))
        print("\n")
        print("Натуралізоване рівняння регресії")
        print("y = {0} + {1}*x1 + {2}*x2\n".format(a0, a1, a2))
        print("Зробимо перевірку:")
        for i in range(len(matr_x2)):
            check = a0 + a1 * matr_x2[i][0] + a2 * matr_x2[i][1]
            print("y{0} = {1}".format(i, check))
        return mean
    else:
        return mean


a = 0
while True:
    b = False
    if inf(a, b):
        n = input("Введіть \"Кінець\" щоб зупинити програму: ")
        if n == "Кінець":
            break
    else:
        n = input("Введіть \"Кінець\" щоб зупинити програму:")
        if n == "Кінець":
            break
        print("Збільшуємо m на 1")
        a += 1