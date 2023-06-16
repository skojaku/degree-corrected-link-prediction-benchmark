# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2023-05-26 10:25:40
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-06-16 11:39:42
# %%
import numpy as np
from os.path import join as j
import itertools
import pandas as pd
from snakemake.utils import Paramspace
import GPUtil
from random import randint
from time import sleep
import string
import itertools
from typing import Dict, Any
import hashlib
import json
from itertools import product
from collections import ChainMap
import gc


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


def to_union_paramspace(param_list):
    df = pd.concat([param2paramDataFrame(l) for l in param_list])
    return Paramspace(df, filename_params="*")


def constrain_by(str_list):
    if len(str_list) == 0:
        return "None"

    print("(" + ")|(".join(str_list) + ")")
    return "(" + ")|(".join(str_list) + ")"


def partial_format(filename, **params):
    field_names = [v[1] for v in string.Formatter().parse(filename) if v[1] is not None]
    fields = {field_name: "{" + field_name + "}" for field_name in field_names}
    for k, v in params.items():
        fields[k] = v
    return filename.format(**fields)


def to_list_value(params):
    for k, v in params.items():
        if isinstance(v, list):
            continue
        else:
            params[k] = [v]
    return params


def expand(filename, *args, **params):
    def _expand(filename, **params):
        paramspace_list = []
        for k, v in params.items():
            if type(v).__name__ != "ParamspaceExtended":
                v = to_paramspace(paramName=k, param={k: v}, index=k)
            paramspace_list.append(v.param_table.to_dict(orient="records"))
        df = pd.DataFrame([dict(ChainMap(*a)) for a in product(*paramspace_list)])
        retval_filename = []
        for d in df.to_dict(orient="records"):
            retval_filename.append(partial_format(filename, **d))

        return retval_filename

    retval = []
    if len(args) == 0:
        return _expand(filename, **params)
    for l in args:
        retval += _expand(filename, **l, **params)
    return retval


def _dict_hash(dictionary) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def to_paramspace(index=None, **params):
    return ParamspaceExtended(**params, filename_params="*", index=index)


class ParamspaceExtended(Paramspace):
    def __init__(
        self,
        paramName,
        param,
        filename_params=None,
        param_sep="~",
        filename_sep="_",
        single_wildcard=None,
        index=None,
    ):
        dflist = []
        if not isinstance(param, list):
            param = [param]
        if not isinstance(index, list) and index is not None:
            index = [index]

        for v in param:
            _params = to_list_value(v.copy())
            keys, values = zip(*_params.items())
            df = pd.DataFrame(
                [dict(zip(keys, bundle)) for bundle in itertools.product(*values)]
            )

            paramDict = df.to_dict(orient="records")
            paramList = []
            for i, r in enumerate(paramDict):
                r = {
                    paramName: _dict_hash(r),
                    "paramValue": r,
                }
                if index is not None:
                    r.update(df[index].iloc[i].to_dict())
                paramList.append(r)
            df = pd.DataFrame(paramList)
            dflist.append(df)
        df = pd.concat(dflist)

        self.param_table = df.drop(columns=["paramValue"])
        self.hashTable = df.rename(columns={paramName: "__hash__"}).set_index(
            "__hash__"
        )
        self.data_table = df.rename(columns={paramName: "hash"})
        self.data_table["paramName"] = paramName

        super(ParamspaceExtended, self).__init__(
            self.param_table,
            filename_params=filename_params,
            param_sep=param_sep,
            filename_sep=filename_sep,
            single_wildcard=single_wildcard,
        )

    def instance(self, wildcards):
        hash_keys = super(ParamspaceExtended, self).instance(wildcards)
        retval = {}
        for paramName, hash in hash_keys.items():
            if hash in self.hashTable.index:
                retval.update(self.hashTable.loc[hash, "paramValue"])
        return retval

def save_param_table(filename):
    firstWrite = True
    for ob in gc.get_objects():
        if not type(ob).__name__ == "ParamspaceExtended":
            continue
        ob.data_table.to_csv(filename, index = False, mode="w" if firstWrite else "a")
        firstWrite = False

#
# emb_params = [
#    {
#        "directed": ["directed", "undirected"],
#        "window_length": [10],
#        "model": ["node2vec", "deepwalk"],
#        "dim": [64, 128],
#    },
#    {
#        "window_length": [5],
#        "model": ["doc2vec"],
#        "dim": [64],
#    },
#    {
#        "model": ["sentence-bert"],
#        "dim": [64, 128],
#    },
# ]
#
# paramspace = to_paramspace(paramName="embedding", param=emb_params, index=["model"])
# EMB_FILE = f"emb_{paramspace.wildcard_pattern}" + "_data~{data}.npz"
# expand(EMB_FILE, data=["aps", "mag"]),
#
#
# paramspace = to_paramspace(
#    paramName="data", param={"data": ["mag", "aps"]}, index="data"
# )
#
