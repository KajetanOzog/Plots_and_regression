x_1 = float(input("Podaj x pierwszego kola "))
y_1 = float(input("Podaj y pierwszego kola "))
r_1 = float(input("Podaj r pierwszego kola "))
x_2 = float(input("Podaj x drugiego kola "))
y_2 = float(input("Podaj y drugiego kola "))
r_2 = float(input("Podaj r drugiego kola "))

d = (x_1 - x_2)**2 + (y_1 - y_2)**2

if (r_1 + r_2)**2 > d:
    print("sa wspolne")
elif (r_1 + r_2)**2 == d:
    print("Styka sie")
else:
    print("brak wspolnych")



