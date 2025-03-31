import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


class Histogram:
    def __init__(self, row: list):
        self.row = np.sort(np.array(row, float))
        self.count_for_small_length = 100

    def __make(self):
        n = len(self.row)
        if n <= self.count_for_small_length:
            number_of_intervals = 1 + int(math.log2(n))
        else:
            number_of_intervals = n // 10
        width_of_intervals = (self.row[-1] - self.row[0]
                              ) / (number_of_intervals - 1)
        start = self.row[0] - width_of_intervals / 2
        partition_interval = [(start + width_of_intervals * i, start +
                               width_of_intervals * (i + 1)) for i in range(number_of_intervals)]
        result = np.zeros(number_of_intervals, float)
        part = 0
        for number in self.row:
            while not (partition_interval[part][0] < number <= partition_interval[part][1]):
                part += 1
            if partition_interval[part][0] < number <= partition_interval[part][1]:
                result[part] += 1
        divider = n * width_of_intervals
        return {"x": [(end - start) / 2 + start for start, end in partition_interval],
                "y": result,
                "divider": divider,
                "width": width_of_intervals}

    def show(self):
        data = self.__make()
        for i in range(len(data["x"])):
            x_row = [data["x"][i] - data["width"] / 2] * 2 + [data["x"][i] + data["width"] / 2] * 2
            y_row = [0, data["y"][i], data["y"][i], 0]
            plt.plot(x_row, y_row, color="black")
        plt.ylabel(f"1/{data['divider']}")
        plt.bar(data["x"], data["y"], width=data["width"])
        plt.show()


class EBF:
    def __init__(self, row: list[float]):
        self.row = sorted(row)

    def __make(self):
        counts = Counter(self.row)
        unique_numbers = sorted(counts.keys())
        cumulative_counts = [0]
        total = 0
        for number in unique_numbers:
            total += counts[number]
            cumulative_counts.append(total)

        n = len(self.row)
        cumulative_counts = [count for count in cumulative_counts]
        unique_numbers = sorted(unique_numbers + [min(unique_numbers) - 1])
        print(unique_numbers)
        print(cumulative_counts)
        return unique_numbers, cumulative_counts, len(self.row)

    def show(self):
        points_x, points_y, ln = self.__make()
        for i in range(len(points_x)):
            if i == 0:
                plt.hlines(y=points_y[i], xmin=points_x[i] - 1, xmax=points_x[i], colors='black')
            else:
                plt.hlines(y=points_y[i - 1], xmin=points_x[i - 1], xmax=points_x[i], colors='black')
        plt.hlines(y=points_y[-1], xmin=points_x[-1], xmax=points_x[-1] + 1, colors='black')
        plt.scatter(points_x[1:], points_y[1:], color="black")
        plt.ylabel(f"1/{ln}")
        plt.grid(True)
        plt.show()
