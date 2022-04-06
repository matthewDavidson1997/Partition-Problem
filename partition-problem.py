from itertools import chain

import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import pandas as pd


def generate_random_array(min_val, max_val, array_length):
    """_summary_

    Args:
        min_val (int):      _description_
        max_val (int):      _description_
        array_length (int): _description_

    Returns:
        list: _description_
    """
    multiset_s = random.randint(min_val, max_val, size=(array_length))
    return multiset_s


def show_partition_percentages(s1, s2, s):
    """_summary_

    Args:
        s1 (list): _description_
        s2 (list): _description_
        s (list): _description_

    Returns:
        string: _description_
    """
    s1_percent = "{:.2%}".format(s1 / s)
    s2_percent = "{:.2%}".format(s2 / s)
    return s1_percent, s2_percent


def compare_output(s1, s2):
    """_summary_

    Args:
        s1 (list): _description_
        s2 (list): _description_

    Returns:
        int: _description_
    """
    if sum(s1) == sum(s2):
        return 0
    elif sum(s1) > sum(s2):
        return (sum(s1) - sum(s2))
    elif sum(s1) < sum(s2):
        return (sum(s2) - sum(s1))


def a_split_array_in_middle(array):
    """_summary_

    Args:
        array (list): _description_

    Returns:
        list: _description_
    """
    s1, s2 = np.array_split(array, 2)

    return s1, s2


def b_split_array_even_and_odd(array):
    """_summary_

    Args:
        array (list): _description_

    Returns:
        list: _description_
    """
    s1 = []
    s2 = []

    for i in array:
        if (i % 2 == 0):
            s1.append(i)
        else:
            s2.append(i)

    return s1, s2


def c_split_array_into_smallest_subarray(array):
    """_summary_

    Args:
        array (list): _description_

    Returns:
        list: _description_
    """
    s1 = []
    s2 = []

    for i in array:
        if sum(s1) <= sum(s2):
            s1.append(i)
        else:
            s2.append(i)

    return s1, s2


def d_split_array_by_mean_then_greedy(array):
    """_summary_

    Args:
        array (list): _description_

    Returns:
        list: _description_
    """
    array_mean = (sum(array) / len(array))
    array_lower = []
    array_higher = []
    s1 = []
    s2 = []
    for i in array:
        if i <= array_mean:
            array_lower.append(i)
        else:
            array_higher.append(i)

    for i in array_higher:
        if sum(s1) <= sum(s2):
            s1.append(i)
        else:
            s2.append(i)

    for i in array_lower:
        if sum(s1) <= sum(s2):
            s1.append(i)
        else:
            s2.append(i)

    return s1, s2


def e_split_array_by_sorted_index(array):
    """_summary_

    Args:
        array (list): _description_

    Returns:
        list: _description_
    """
    s1 = []
    s2 = []
    sorted_array_s = np.sort(array)

    for i in range(0, len(sorted_array_s)):
        if i % 2 == 0:
            s1.append(sorted_array_s[i])
        else:
            s2.append(sorted_array_s[i])

    return s1, s2


def f_split_array_adding_to_smallest_subarray_after_descending_sort(array):
    """_summary_

    Args:
        array (list): _description_

    Returns:
        list: _description_
    """
    s1 = []
    s2 = []
    reverse_sorted_array_s = -np.sort(-array)

    for i in reverse_sorted_array_s:
        if sum(s1) <= sum(s2):
            s1.append(i)
        else:
            s2.append(i)
    return s1, s2


def run_tests(cardinality):
    """_summary_

    Args:
        cardinality (int): _description_

    Returns:
        int: _description_
    """
    multiset_s = generate_random_array(1, cardinality * 10, cardinality)

    a_s1, a_s2 = a_split_array_in_middle(multiset_s)
    a_diff = compare_output(a_s1, a_s2)
    b_s1, b_s2 = b_split_array_even_and_odd(multiset_s)
    b_diff = compare_output(b_s1, b_s2)
    c_s1, c_s2 = c_split_array_into_smallest_subarray(multiset_s)
    c_diff = compare_output(c_s1, c_s2)
    d_s1, d_s2 = d_split_array_by_mean_then_greedy(multiset_s)
    d_diff = compare_output(d_s1, d_s2)
    e_s1, e_s2 = e_split_array_by_sorted_index(multiset_s)
    e_diff = compare_output(e_s1, e_s2)
    f_s1, f_s2 = f_split_array_adding_to_smallest_subarray_after_descending_sort(multiset_s)
    f_diff = compare_output(f_s1, f_s2)

    return a_diff, b_diff, c_diff, d_diff, e_diff, f_diff, multiset_s


def mean_absolute_deviation(array):
    """_summary_

    Args:
        array (list): _description_

    Returns:
        int: _description_
    """
    mad_result_series = pd.Series(array)
    mad_result = mad_result_series.mad()

    return mad_result


def run_mean_deviations(a_results, b_results, c_results, d_results, e_results, f_results, cardinality):
    """_summary_

    Args:
        a_results (list): _description_
        b_results (list): _description_
        c_results (list): _description_
        d_results (list): _description_
        e_results (list): _description_
        f_results (list): _description_
        cardinality (int): _description_

    Returns:
        list: _description_
    """
    a = mean_absolute_deviation(a_results)
    b = mean_absolute_deviation(b_results)
    c = mean_absolute_deviation(c_results)
    d = mean_absolute_deviation(d_results)
    e = mean_absolute_deviation(e_results)
    f = mean_absolute_deviation(f_results)

    mean_deviations_y = [a, b, c, d, e, f]
    mean_deviations_x = [str(cardinality)] * len(mean_deviations_y)

    return mean_deviations_x, mean_deviations_y


def main():
    cardinalities_to_test = [32, 64, 128, 256, 512, 1024]
    number_of_tests = 10000

    a_results = []
    b_results = []
    c_results = []
    d_results = []
    e_results = []
    f_results = []
    mean_deviations_x = []
    mean_deviations_y = []
    standard_deviations = []

    for c in cardinalities_to_test:
        for i in range(number_of_tests):
            a_diff, b_diff, c_diff, d_diff, e_diff, f_diff, multiset_s = run_tests(c)
            a_results.append(a_diff)
            b_results.append(b_diff)
            c_results.append(c_diff)
            d_results.append(d_diff)
            e_results.append(e_diff)
            f_results.append(f_diff)
            mean_deviations_results_x, mean_deviations_results_y = run_mean_deviations(a_results, b_results, c_results, d_results, e_results, f_results, c)
            standard_deviation_array = np.array(multiset_s)
            standard_deviation = np.std(standard_deviation_array)
            print(i, c)
        standard_deviations.append(standard_deviation)
        mean_deviations_x.append(mean_deviations_results_x)
        mean_deviations_y.append(mean_deviations_results_y)
        
        

    results_table = pd.DataFrame(data=mean_deviations_y, index=['32', '64', '128', '256', '512', '1024'], columns=['A', 'B', 'C', 'D', 'E', 'F'])
    print(results_table)
    print(len(standard_deviations))
    results_table.plot(kind='line', marker='+')
    plt.title('Mean deviation of partitions:')
    # plt.errorbar(list(chain(*mean_deviations_x)), list(chain(*mean_deviations_y)), standard_deviations)
    plt.xlabel('Cardinality')
    plt.ylabel('Average deviation from mean')
    plt.yscale('log')
    # plt.legend()
    plt.show()

    # results_table.to_csv('results_table.csv')

    print(mean_deviations_x)
    print(mean_deviations_y)
    print(standard_deviations)

if __name__ == '__main__':

    main()

    
