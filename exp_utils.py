import json
import pprint
#from haven import haven_utils as hu
import os

"""
def _filter_fn(val):
    if val == "true":
        return True
    elif val == "false":
        return False
    elif val == "none":
        return None
    else:
        return val
"""

def _filter_fn(val):
    if type(val) is list:
        val = [ _filter_fn(elem) for elem in val ]
        return val
    else:
        if val == "true":
            return True
        elif val == "false":
            return False
        elif val == "none":
            return None
        elif type(val) is str:
            # Will do expansion on environment variables
            return os.path.expandvars(val)
        else:
            return val


def _traverse(dict_, keys, val):
    if len(keys) == 0:
        return
    else:
        # recurse
        if keys[0] not in dict_:
            if len(keys[1:]) == 0:
                dict_[keys[0]] = _filter_fn(val)
            else:
                dict_[keys[0]] = {}
        _traverse(dict_[keys[0]], keys[1:], val)

def unflatten(dict_):
    new_dict = {}
    for key, val in dict_.items():
        key_split = key.split(".")
        _traverse(new_dict, key_split, val)
    return new_dict

def enumerate_and_unflatten(filename):
    dict_ = json.loads(open(filename).read())
    exps = hu.cartesian_exp_group(dict_)
    return [unflatten(dd) for dd in exps]