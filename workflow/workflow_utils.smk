import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
from random import randint
from time import sleep
import string
import itertools

# Utilities
def param2paramDataFrame(param_list):
    if isinstance(param_list, list) is False:
        param_list = [param_list]
    my_dict = {}
    cols = []
    for dic in param_list:
        my_dict.update(dic)
        cols += list(dic.keys())
    keys, values = zip(*my_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    df = pd.DataFrame(permutations_dicts)
    df = df[cols]
    return df


def to_grid_paramspace(param_list):
    df = param2paramDataFrame(param_list)
    return Paramspace(df, filename_params="*")


def to_union_paramspace(param_list):
    df = pd.concat([param2paramDataFrame(l) for l in param_list])
    return Paramspace(df, filename_params="*")


def constrain_by(str_list):
    return "(" + ")|(".join(str_list) + ")"

def partial_format(filename, **params):
    field_names = [v[1] for v in string.Formatter().parse(filename) if v[1] is not None]
    fields = {field_name:"{"+field_name+"}" for field_name in field_names}
    for k,v in params.items():
        fields[k] = v
    return filename.format(**fields)

def to_list_value(params):
    for k, v in params.items():
        if isinstance(v, list):
            continue
        else:
            params[k]=[v]
    return params

def _expand(filename, **params):
    params = to_list_value(params)
    retval_filename = []
    keys, values = zip(*params.items())
    for bundle in itertools.product(*values):
        d = dict(zip(keys, bundle))
        retval_filename.append(partial_format(filename, **d))
    return retval_filename

def expand(filename, *args, **params):
    retval = []
    if len(args) == 0:
        return _expand(filename, **params)
    for l in args:
        retval += _expand(filename, **l, **params)
    return retval
