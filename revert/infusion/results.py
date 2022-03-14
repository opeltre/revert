import os
import json
import pandas as pd

PREFIX = (os.environ["INFUSION_DATASETS"] 
            if "INFUSION_DATASETS" in os.environ
            else os.getcwd())

#--- Physiological bounds ---

BOUNDS = {
    "Elastance [1/ml]"              : [.05, 5],
    "Pss [mmHg]"                    : [-5, "ICP baseline [mmHg]"],
    "CSF production rate [ml/min]"  : [.1, 1]
}

class Results :
    """ Interface to the table of ICM+ computed results """

    def __init__(self, path):
        """ Load json data from $INFUSION_DATASETS/<path> """
        self.path = (os.path.join(PREFIX, path) if path[0] != '/' else path)
        res = self.json()
        val, index = [], []
        for k, rk in res.items():
            val += [rk]
            index += [k]
        self.dataframe = pd.DataFrame(val, index=index)

    def json(self): 
        """ Returned the dictionary of parsed json data."""
        with open(self.path, "r") as f:
            data = json.load(f)
        return data

    def bounded (self, bounds=BOUNDS):
        """ Check whether values are within physiological bounds."""
        df = self.dataframe
        series = []
        for k, bs in bounds.items():
            b0, b1 = [df[b] if type(b) == str else b for b in bs]
            series += [df[k].between(b0, b1)]
        return pd.concat(series, axis=1).all(axis=1)

    def shunted (self, val=True):
        """ Check whether a shunt is being tested."""
        field = "Shunt resist. [mmHg*min/ml]"
        df    = self.dataframe
        return df[field].notnull() if val else df[field].isnull()

    def filtered (self, bounds=BOUNDS, shunted=False):
        """ Return a filtered dataframe of unshunted patients within bounds."""
        s1 = self.bounded(bounds)
        s2 = self.shunted(shunted)
        idx = pd.concat([s1, s2], axis=1).all(axis=1)
        return self.dataframe[idx]
