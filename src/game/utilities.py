import random


def random_bool(true_probability=0.5):
    return random.choices(
        [True, False],
        weights=[true_probability, 1.0 - true_probability]
    )[0]
