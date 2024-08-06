import numpy as np


class Generator:

    def __init__(self): pass


class BinaryKnapsackGenerator(Generator):

    def __init__(self, n_items: int, coefficient_range: int, instances_in_series: int = 1000):

        super().__init__()

        assert n_items > 0
        assert coefficient_range > 0
        assert instances_in_series > 0

        self.n_items = n_items
        self.coefficient_range = int(coefficient_range)
        self.instances_in_series = instances_in_series

        self.last_instance = {"values": np.zeros(n_items), "weights": np.zeros(n_items), "capacity": 0.0}
        self.h48 = None
        self.l48 = None

    def get_coefficient_range(self):

        return self.coefficient_range

    def srand(self, s):

        self.h48 = s
        self.l48 = 0x330E

    def lrand(self):

        self.h48 = (self.h48 * 0xDEECE66D) + (self.l48 * 0x5DEEC)
        self.l48 = self.l48 * 0xE66D + 0xB
        self.h48 = self.h48 + (self.l48 >> 16)
        self.l48 = self.l48 & 0xFFFF
        return self.h48 >> 1

    def randm(self, x):

        return self.lrand() % int(x)

    def generate(self, instance_type: str, instance_number: int, normalize: bool = True):

        assert instance_type in {"uncorrelated", "weakly-correlated", "strongly-correlated", "subset-sum"}
        assert 0 <= instance_number <= 1000

        self.srand(instance_number)
        coefficient_range_1 = self.coefficient_range / 10

        for i in range(self.n_items):

            value = 0
            weight = self.randm(self.coefficient_range) + 1

            if instance_type == "uncorrelated":
                value = self.randm(self.coefficient_range) + 1
            elif instance_type == "weakly-correlated":
                value = min(self.coefficient_range,
                            max(0, self.randm(2 * coefficient_range_1 + 1) + weight - coefficient_range_1))
            elif instance_type == "strongly-correlated":
                value = weight + 10
            elif instance_type == "subset-sum":
                value = weight

            self.last_instance["values"][i] = value
            self.last_instance["weights"][i] = weight

        capacity = (instance_number * self.last_instance["weights"].sum()) / (self.instances_in_series + 1)
        if capacity <= self.coefficient_range:
            capacity = self.coefficient_range + 1

        self.last_instance["capacity"] = capacity

        if normalize:
            self.last_instance["values"] /= self.coefficient_range
            self.last_instance["weights"] /= self.last_instance["capacity"]
            self.last_instance["capacity"] = 1.0

        return self.last_instance
