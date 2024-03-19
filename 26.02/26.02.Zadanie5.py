def f(lista):
    il = 1
    s = 0
    for i in lista:
        il *= i
        s += i

    r = lista[0]
    for i in range(1, len(lista)):
        r -= lista[i]
    return s, r, il


l = [1, 2, 3, 4, 5, 6, 7]
d = [5, 1, 1, 1, 1]
print(f(l))
print(f(d))
