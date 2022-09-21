import numpy as np

FLOAT_RELATIVE_TOL = 1e-6


def compare_floats(number1: float, number2: float):
    return np.isclose(number1, number2, rtol=FLOAT_RELATIVE_TOL)


def compare_dictionary(dict1: dict, dict2: dict):
    if dict1.keys() != dict2.keys():
        return False
    for key, value in dict1.items():
        if type(value) is str:
            if value != dict2[key]:
                return False
        if type(value) is int:
            if value != dict2[key]:
                return False
        if type(value) is float:
            if not compare_floats(value, dict2[key]):
                return False
        if value != dict2[key]:
            return False
    return True
