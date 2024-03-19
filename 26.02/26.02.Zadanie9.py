import random
random.seed()


def monte_carlo_Pi(promien, ilosc_losowan):
    trafione = 0
    for i in range(ilosc_losowan):
        x = random.randint(-1*promien, promien)
        y = random.randint(-1*promien, promien)
        if x**2 + y**2 <= promien**2:
            trafione += 1
    return 4 * (trafione/ilosc_losowan)


print(monte_carlo_Pi(100, 1000000))
