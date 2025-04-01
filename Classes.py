import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


class R1Z1:
    def __init__(self, path_csv: str):
        with open(path_csv, 'r') as file:
            file.readline()
            self.row = sorted([float(number) for number in file])

    def __make_Histogram(self):
        n = len(self.row)
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

    def load_Histogram(self, path):
        plt.clf()
        data = self.__make_Histogram()
        for i in range(len(data["x"])):
            x_row = [data["x"][i] - data["width"] / 2] * \
                2 + [data["x"][i] + data["width"] / 2] * 2
            y_row = [0, data["y"][i], data["y"][i], 0]
            plt.plot(x_row, y_row, color="black")
        plt.ylabel(f"1/{data['divider']}")
        plt.bar(data["x"], data["y"], width=data["width"])
        plt.savefig(path)

    def __make_EBF(self):
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
        return unique_numbers, cumulative_counts, len(self.row)

    def load_EBF(self, path):
        plt.clf()
        points_x, points_y, ln = self.__make_EBF()
        for i in range(len(points_x)):
            if i == 0:
                plt.hlines(y=points_y[i], xmin=points_x[i] -
                           1, xmax=points_x[i], colors='black')
            else:
                plt.hlines(
                    y=points_y[i - 1], xmin=points_x[i - 1], xmax=points_x[i], colors='black')
        plt.hlines(y=points_y[-1], xmin=points_x[-1],
                   xmax=points_x[-1] + 1, colors='black')
        plt.scatter(points_x[1:], points_y[1:], color="black")
        plt.ylabel(f"1/{ln}")
        plt.grid(True)
        plt.savefig(path)

    def calculate_percentile(self, p):
        k = (len(self.row) - 1) * p
        return (self.row[int(k)] + self.row[int(k) + 1]) / 2 if k != int(k) else self.row[k]

    def get_median(self):
        count = len(self.row)
        return self.row[(count - 1) // 2] if count % 2 == 0 else (
            self.row[(count - 1) // 2] + self.row[(count - 1) // 2 + 1]) / 2

    def reformat(func):
        def wrapper(self, *args, **kwargs):
            stats = func(self)
            List = [f"{key}: {item}" for key, item in stats.items()]
            print('\n'.join(List))
        return wrapper

    @reformat
    def print_stat(self):
        count = len(self.row)
        mean = sum(self.row) / len(self.row)
        min_ = self.row[0]
        max_ = self.row[-1]
        pvariance = sum((i - mean) ** 2 for i in self.row) / count
        asymmetry = 1 / (count * pvariance ** 1.5) * \
            sum((i - pvariance) ** 3 for i in self.row)
        median = self.get_median()
        interquartile_range = self.calculate_percentile(3/4) - \
            self.calculate_percentile(1/4)
        stats = {
            "count": count,
            "min": min_,
            "max": max_,
            "range": max_ - min_,
            "mean": mean,
            "pvariance": pvariance,
            "variance": count / (count - 1) * pvariance,
            "deviation": math.sqrt(pvariance),
            "asymetry": asymmetry,
            "median": median,
            "interquartile_range": interquartile_range
        }
        return stats
