def reverse(napis):
    nowy_napis = ""
    for i in range(len(napis) - 1, -1, -1):
        nowy_napis += napis[i]
    return nowy_napis


napis = input("Podaj napis: ")
print(napis)
print(reverse(napis))
