import random
from collections import ChainMap


def normalize_dataset(db):
    if db["meta"]["ethnicities"]:
        perms = [e + g for e in db["meta"]["ethnicities"]
                 for g in db["meta"]["genders"]]
    else:
        perms = db["meta"]["genders"]
    db_dict_list = [db["data"][key.lower()] for key in perms]
    len_min = min([len(el) for el in db_dict_list])
    db_dict_list_norm = [{key: el[key] for key in random.sample(
        list(el), len_min)} for el in db_dict_list]
    data = {key: db_dict_list_norm[i] for i, key in enumerate(perms)}
    data.update(dict(
        all=ChainMap(*db_dict_list_norm),
        train=db["data"]["train"],
        test=db["data"]["test"]))
    print(len(data["all"].keys()))
    return dict(data=data, meta=db["meta"])
