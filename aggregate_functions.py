import numpy as np


def get_support_agg_len(dict_inter):
    agg_len = {}
    for key, value in dict_inter.items():
        if agg_len.get(len(key)) is not None:
            agg_len[len(key)] = agg_len[len(key)] + np.array(value)
        else:
            agg_len[len(key)] = np.array(value)
    return agg_len


def simple_aggregate_function(dict_inter):
    sum_plus = 0
    sum_neg = 0
    for key, value in dict_inter.items():
        sum_plus += value[0]
        sum_neg += value[1]
    # set_var = set()
    # for k in sorted(dict_inter, key=len, reverse=True):  # Through keys sorted by length
    #     if set(k) <= set(set_var):
    #         # print('kek')
    #         # print(k)
    #         # print(set_var)
    #         continue
    #     v = dict_inter[k]
    #     sum_plus += v[0]
    #     sum_neg += v[1]
    #     set_var.update(k)
    return int(sum_plus >= sum_neg), sum_plus, sum_neg


def simple_aggregate_function_ps(dict_inter):  # ps - pattern structures
    sum_plus = 0
    sum_neg = 0
    sum_plus_cf = 0
    sum_neg_cf = 0
    for key, value in dict_inter.items():
        sum_plus += value[0]
        sum_neg += value[1]
        sum_plus_cf += value[2]
        sum_neg_cf += value[3]
    return int(sum_plus + sum_plus_cf >= sum_neg + sum_neg_cf), sum_plus, sum_neg, sum_plus_cf,  sum_neg_cf


def simple_exp_agg_predict(dict_inter, alpha=0.3):
    tot_sum = 0
    for key, value in dict_inter.items():
        if np.abs(value[0] - value[1]) > alpha:
            tot_sum += value[0] - value[1]
        # sum_plus += value[0]
        # sum_neg += value[1]
    return int(tot_sum > 0), tot_sum


def simple_weight_aggregate_function(dict_inter):
    sum_plus = 0
    sum_neg = 0
    for key, value in dict_inter.items():
        sum_plus += len(key) * value[0]
        sum_neg += len(key) * value[1]
    return int(sum_plus > sum_neg), sum_plus, sum_neg
