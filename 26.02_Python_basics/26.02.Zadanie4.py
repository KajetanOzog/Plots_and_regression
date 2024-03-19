d = int(input("Podaj dlugosc: "))
print("|" + "...|"*d)
for i in range(d+1):
    print("%-4d" % i, end="")
