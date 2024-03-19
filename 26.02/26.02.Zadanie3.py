n = int(input())
m = int(input())

a = []

for i in range(0, n):
    b = []
    if i % 2 == 0:
        for j in range(0, m):
            b.append(j % 2)
    else:
        for j in range(0, m):
            b.append(((j + 1) % 2))
    a.append(b)

for i in a:
    for j in i:
        print(j, end="")
    print()


