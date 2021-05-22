from random import randrange
from prettytable import PrettyTable
import time

table0 = PrettyTable()
table0.field_names = (["Студент", "Группа"])
name = "Карнаухова Анастасія"
group = "ІО-92"
table0.add_row([name, group])
print(table0)

a0 = int(input("Введіть значення a0: "))
a1 = int(input("Введіть значення a1: "))
a2 = int(input("Введіть значення a2: "))
a3 = int(input("Введіть значення a3: "))

x1 = [randrange(1,21) for _ in range(8)]
x2 = [randrange(1,21) for _ in range(8)]
x3 = [randrange(1,21) for _ in range(8)]

x01 = (max(x1)+min(x1))/2
x02 = (max(x2)+min(x2))/2
x03 = (max(x3)+min(x3))/2

dx1 = x01-min(x1)
dx2 = x02-min(x2)
dx3 = x03-min(x3)

xn1 = [round((x1[i] - x01)/dx1, 2) for i in range(8)]
xn2 = [round((x2[i] - x02)/dx2, 2) for i in range(8)]
xn3 = [round((x3[i] - x03)/dx3, 2) for i in range(8)]

list_y = [a0 + a1*x1[i] + a2*x2[i] + a3*x3[i] for i in range(8)]
Y_et = a0 + a1*x01 + a2*x02 + a3*x03

result = max(list_y)

print("\nТаблиця:")
table_main = [x1, x2, x3, list_y, xn1, xn2, xn3]
table_value = list(zip(*table_main))
table = PrettyTable()
table.field_names = ["#", "x1", "x2", "x3", "Y", "хн1", "хн2", "хн3"]

for x in range(1, len(table_value)+1):
    table.add_row([x] + list(table_value[x-1]))

table.add_row(["x0", x01, x02, x03, "", "", "", ""])
table.add_row(["dx", dx1, dx2, dx3, "", "", "", ""])

print(table)

print("Ymax: {0}".format(result))
