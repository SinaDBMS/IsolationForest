from collections.abc import Iterator
from itertools import product
from typing import List, Dict


class ParameterGrid(Iterator):

    def __init__(self, params: Dict):
        self.__raw_parameters: Dict = params
        self.__params: List[Dict] = []
        self.__position = 0
        keys = params.keys()
        for values in product(*params.values()):
            self.__params.append(dict(zip(keys, values)))

    def __next__(self):
        if self.__position >= len(self.__params):
            raise StopIteration()
        value = self.__params[self.__position]
        self.__position += 1
        return value

    @property
    def params(self) -> Dict:
        return self.__raw_parameters

    @property
    def size(self):
        return len(self.__params)
