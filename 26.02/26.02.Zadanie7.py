napis = input("Podaj napis: ")
print(napis)

nowy_napis = ""

for i, x in enumerate(napis):
    if i == 0 or i == len(napis):
        nowy_napis += x
    else:
        nowy_napis += " {}".format(x)

print(nowy_napis)
