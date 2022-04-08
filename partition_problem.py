import random

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


class ArraySplitter:
    """A class to split an array using different named splitting methods.

    Methods:
        `split()`: splits the array based on a named method, and returns the two subsets.
        `compare_methods()`: Applies all splitting methods. Returns the absolute subset sum difference
                             for each method as a dictionary.
    """
    def __init__(self, array):
        """Initialise an ArraySplitter instance with an input array.

        Args:
            array (list): A multi-set integer array (values may be duplicated).
        """
        self.array = array
        # Initialise a dictionary we can use to get method names and complexities from
        self.func_dict = {
            'A': {'method': '_split_middle', 'complexity': 'O(n)'},
            'B': {'method': '_split_even_odd', 'complexity': 'O(n)'},
            'C': {'method': '_split_greedy', 'complexity': 'O(n)'},
            'D': {'method': '_split_mean_greedy', 'complexity': 'O(n)'},
            'E': {'method': '_split_sort_index', 'complexity': 'O(nlogn)'},
            'F': {'method': '_split_reverse_greedy', 'complexity': 'O(nlogn)'},
            'G': {'method': '_split_kk', 'complexity': 'O(nlogn)'}
        }

    def split(self, name):
        """Split the array using a chosen named method.

        Args:
            name (str): The splitting method.
                'A' - calls the _split_middle() method
                'B' - calls the _split_even_odd() method
                'C' - calls the _split_greedy() method
                'D' - calls the _split_mean_greedy() method
                'E' - calls the _split_sort_index() method
                'F' - calls the _split_reverse_greedy() method

        Returns:
            (list, list): The array split into two sub-arrays.
        """
        # Get the required function by name
        func = getattr(self, self.func_dict[name]['method'])
        return func(self.array)

    # Approach A - time complexity O(n)
    def _split_middle(self, array):
        """Split in the middle of the array
        (If the array length is odd, then s1 is the shorter subset)."""
        s1 = []
        s2 = []
        middle = len(array) // 2  # O(1)
        for idx, val in enumerate(array):  # O(n)
            if idx < middle:  # O(1)
                s1.append(val)
            else:
                s2.append(val)
        return s1, s2

    # Approach B - time complexity O(n)
    def _split_even_odd(self, array):
        """Even numbers go into sub-array 1, odd numbers into sub-array 2."""
        s1 = []
        s2 = []
        for val in array:  # O(n)
            if val % 2 == 0:  # O(1)
                s1.append(val)
            else:
                s2.append(val)

        return s1, s2

    # Approach C - time complexity O(n)
    def _split_greedy(self, array):
        """Partitions the array by adding each member to the set with the smallest sum.
        If there is a tie, then the member is added to the first set.
        """
        s1 = []
        s2 = []
        s1_sum = 0
        s2_sum = 0
        for val in array:  # O(n)
            if s1_sum <= s2_sum:  # O(1)
                s1.append(val)
                s1_sum += val  # O(1)
            else:
                s2.append(val)
                s2_sum += val  # O(1)
        return s1, s2

    # Approach D - time complexity O(n)
    def _split_mean_greedy(self, array):
        """First splits into two sets (members < mean) and (members >= mean).
        The 'lower' subset is appended to the 'higher' subset, and this combined array is processed using the
        'greedy' method.
        """
        array_mean = (sum(array) / len(array))  # O(n) + O(1) + O(1)
        array_lower = []
        array_higher = []
        for val in array:  # O(n)
            if val <= array_mean:  # O(1)
                array_lower.append(val)
            else:
                array_higher.append(val)
        array_total = array_higher + array_lower  # O(n)
        s1, s2 = self._split_greedy(array_total)  # O(n)
        return s1, s2

    # Approach E - time complexity O(nlogn)
    def _split_sort_index(self, array):
        """Sorts the array, then assigns into subset 1 if the index is even, or subset 2 if
        the index is odd."""
        sorted_array = sorted(array, reverse=True)  # O(nlogn) - see https://en.wikipedia.org/wiki/Timsort
        s1 = []
        s2 = []
        for idx, val in enumerate(sorted_array):  # O(n)
            if idx % 2 == 0:  # O(1)
                s1.append(val)
            else:
                s2.append(val)
        return s1, s2

    # Approach F - time complexity O(nlogn)
    def _split_reverse_greedy(self, array):
        """Sorts high to low, then applies the 'greedy' method."""
        reverse_sorted_array = sorted(array, reverse=True)  # O(nlogn)
        s1, s2 = self._split_greedy(reverse_sorted_array)  # O(n)
        return s1, s2

    # Approach G - time complexity O(nlogn)
    def _split_kk(self, array):
        """Sorts high to low, then removes the first two members of the array.
        The 2nd value is subtracted from the first value, and the difference is appended to the end of the list.
        The list is then resorted, and the process is repeated until the list length is 1.

        Finally the list is passed back as s1_, and an empty list is passed back as s2_.
        This is a different approach to the other algorithms, hence the different variable naming for clarity.

        Passing back the two lists allows downstream calculation of the absolute subset sum difference (even though
        we already know the answer)
        """
        s1_ = []
        s2_ = []

        reverse_sorted_array = sorted(array, reverse=True)  # O(nlogn)
        while len(reverse_sorted_array) > 1:  # O(n)
            elem_0 = reverse_sorted_array.pop(0)
            elem_1 = reverse_sorted_array.pop(0)
            reverse_sorted_array.append(elem_0 - elem_1)
            reverse_sorted_array.sort(reverse=True)  # O(nlogn)
        # We know the difference at this point, but we are not using or returning it
        s1_ = reverse_sorted_array
        return s1_, s2_

    def calc_subset_diff(self, s1, s2):
        """Simply calculates the absolute difference between two subsets."""
        return abs(sum(s1) - sum(s2))

    def _print_run_info(self, name, s1, s2, diff):
        """Simply prints some info."""
        print(f'Method: {name}')
        complexity = self.func_dict[name]['complexity']
        print(f'Time complexity: {complexity}')
        print(f'  Subset 1: {s1}')
        print(f'  Subset 2: {s2}')
        print(f'  Absolute partition difference: {diff}')
        print()

    def compare_methods(self, verbose=False):
        """Runs all splitting methods and returns a dictionary of method names against subset sum differences.

        Args:
            verbose (bool, optional): Print information about the splits and subset sum differences.
                                      Defaults to False.

        Returns:
            dict: A dictionary of split method names and absolute subset sum differences.
        """
        performance_dict = {}
        for name in self.func_dict.keys():
            s1, s2 = self.split(name)
            abs_sum_diff = self.calc_subset_diff(s1, s2)
            performance_dict[name] = abs_sum_diff
            if verbose:
                self._print_run_info(name, s1, s2, abs_sum_diff)
        return performance_dict


def generate_random_array(min_val, max_val, cardinality):
    """Generates an array (multiset) of random integers.

    Args:
        min_val (int): The minimum integer value to include in the array.
        max_val (int): The maximum integer value to include in the array.
        cardinality (str): The cardinality (length) of the array.

    Returns:
        list[ints]: A list of random integers.
    """
    return [random.randint(min_val, max_val) for _ in range(cardinality)]


def run_tests(cardinalities, repeats, min_array_val, max_val_factor):
    """Runs repeated tests on each provided cardinality, generating a random array
    each time and comparing each splitting method on that array.

    Args:
        cardinalities (list): Cardinalities to use.
        repeats (int): Number of repeats.
        min_array_val (int): Minimum possible value in the array.
        max_val_factor (int): Scaling factor to set the maximum possible
                              value in the array by multiplying by the cardinality.

    Returns:
        list[dicts]: A list of dictionaries of length 'repeats'.
                     Keys are the method names, plus 'cardinality'.
                     Values are the method absolute subset differences, plus cardinality
    """
    testing_results = []
    for cardinality in cardinalities:
        for _ in range(repeats):
            array = generate_random_array(min_val=min_array_val,
                                          max_val=max_val_factor * cardinality,
                                          cardinality=cardinality)
            splitter = ArraySplitter(array)
            results = splitter.compare_methods()
            results['cardinality'] = cardinality
            testing_results.append(results)
    return testing_results


def seaborn_plot(results, repeats, figsize=(12, 9), facecolor='white', confidence_interval=95):
    """Generate a Seaborn plot of the results data.

    Args:
        results (list[dicts]): A list of results dictionaries - one dictionary per test run.
        repeats (int): The number of repeats employed to generate 'results'.
        figsize (tuple, optional): The figure dimensions in inches. Defaults to (12, 9).
        facecolor (str, optional): The figure background colour. Defaults to 'white'.
        confidence_interval (int or str, optional): Confidence interval percent value to use in the plot.
                                                    Alternatively, if 'sd' the standard deviation will be shown.
                                                    Defaults to 95.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    # Make a dataframe from the list of dictionaries
    testing_df = pd.DataFrame(results)
    # Unpivot all columns not specified in the id_vars list
    tall_df = pd.melt(testing_df, id_vars=['cardinality'], var_name='method', value_name='absolute_diff')
    # Make the plot
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax = sns.lineplot(data=tall_df, x='cardinality', y='absolute_diff', hue='method', ci=confidence_interval)
    ax.set_xlabel('multiset cardinality')
    ax.set_xscale('log', base=2)
    ax.set_ylabel('mean absolute partition difference')
    ax.set_yscale('log')
    # Put legend outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # Underscore assignment to supress Text object output
    _ = ax.set_title(f'Comparison of multiset splitting methods ({repeats} repeats)')
    return fig


def pandas_plot(results, repeats, figsize=(12, 9), facecolor='white'):
    # Make a dataframe from the list of dictionaries
    testing_df = pd.DataFrame(results)
    # Unpivot all columns not specified in the id_vars list
    tall_df = pd.melt(testing_df, id_vars=['cardinality'], var_name='method', value_name='absolute_diff')
    # Group by cardinality and method - sort=False ensures we don't change the method order to alphabetical
    group = tall_df.groupby(['cardinality', 'method'], as_index=False, sort=False)
    # Now aggregate to compute the mean and standard deviation for each cardinality/method group
    # Dropping level 0 of axis 1 allows us to use 'mean' and 'std' as the column names
    # instead of ('absolute_diff', 'mean') and ('absolute_diff', 'std')
    stats_df = group.agg(['mean', 'std']).droplevel(axis=1, level=0)
    # Make the plot
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    # Iterate over each method and add the data to the plot
    for key, group in stats_df.groupby('method'):
        group.reset_index(inplace=True)
        group.plot('cardinality', 'mean', yerr='std', label=key, ax=ax)
    ax.set_xlabel('multiset cardinality')
    ax.set_xscale('log', base=2)
    ax.set_ylabel('mean absolute partition difference')
    ax.set_yscale('log')
    # Put legend outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # Underscore assignment to supress Text object output
    _ = ax.set_title(f'Comparison of multiset splitting methods ({repeats} repeats)')
    return fig


def main():
    """Main method to demonstrate functionality."""
    # Extra import (to catch stdout to a file)
    from contextlib import redirect_stdout

    # Testing parameters
    CARDINALITIES = [32, 64, 128, 256, 512, 1024]
    REPEATS = 100
    MIN_ARRAY_VAL = 1
    MAX_VAL_FACTOR = 10

    # Plotting preference
    sns.set_context('talk')

    # Walkthrough examples with first cardinality value (export data as .txt)
    array = generate_random_array(min_val=MIN_ARRAY_VAL,
                                  max_val=MIN_ARRAY_VAL*MAX_VAL_FACTOR,
                                  cardinality=CARDINALITIES[0])

    splitter = ArraySplitter(array)
    with open('results/walkthrough_examples.txt', 'w') as f:
        with redirect_stdout(f):
            splitter.compare_methods(verbose=True)

    # Apply each splitting method (export raw data as .csv)
    testing_results = run_tests(cardinalities=CARDINALITIES,
                                repeats=REPEATS,
                                min_array_val=MIN_ARRAY_VAL,
                                max_val_factor=MAX_VAL_FACTOR)

    testing_df = pd.DataFrame(testing_results)
    testing_df.to_csv('results/testing_df.csv')

    # Make seaborn plot
    fig = seaborn_plot(testing_results, repeats=REPEATS)
    fig.savefig('results/comparison_seaborn.png')

    # Make pandas plot
    fig = pandas_plot(testing_results, repeats=REPEATS)
    fig.savefig('results/comparison_pandas.png')


if __name__ == '__main__':
    main()
