

class BetDistribution:

    def distribute_double_coverage_smallest_diff(self, probabilities: list[dict], num_double_coverages=7):
        differences = []
        for n, prob in enumerate(probabilities):
            values = list(prob.values())
            max_val = max(values)
            second_largest = sorted(values)[1]
            val_index = values.index(second_largest)

            differences.append({
                'index': n,
                'value': list(prob.keys())[val_index],
                'difference': max_val - second_largest,
            })

        sorted_differences = sorted(
            differences,
            key=lambda item: item["difference"],
        )

        return sorted_differences[:num_double_coverages]

    def distribute_double_coverage_biggest_coverage(self, probabilities: list[dict], num_double_coverages=7):
        sums = []
        for n, prob in enumerate(probabilities):
            values = sorted(list(prob.values()))

            sums.append({
                'index': n,
                'value': values[1],
                'sum': values[1] + values[2],
            })

        sorted_sums = sorted(
            sums,
            key=lambda item: item["sum"],
        )

        return sorted_sums[:num_double_coverages]






