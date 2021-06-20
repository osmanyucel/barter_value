import math


def get_sigmoid(val):
    return (1 / (1 + math.exp(-val)) - 0.5) * 2