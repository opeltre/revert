import os
import re
import json

from .file import File

PREFIX = (os.environ["INFUSION_DATASETS"]
            if "INFUSION_DATASETS" in os.environ
            else os.getcwd())

class Dataset:
    """ 
    Interface to HDF5 directories of continuous ICP recordings.

    Usage:
        - access specific or random HDF5 files : `dset.get(pattern)`,
        - yield ICP slices of constant length  : `iter(dataset)`,
        - safe mapping/filtering of metadata   : `dset.map`, `dset.filter`.
    """

    def __init__(self, path='no_shunt', timestamp=None, minutes=4, fs=100):
        """ 
        Dataset reading ICP files from `path`. 

        The directory path can be absolute or relative to `$INFUSION_DATASETS`.
        If provided, `timestamp` and `minutes` are used by `iter(dset)` to 
        yield adequate slices of the ICP signal, safely ignoring files with missing 
        metadata.
        """
        self.name = os.path.basename(path)
        self.path = (path if path[0] == "/"
                        else os.path.join(PREFIX, path))
        #--- Set to 'Baseline', 'Plateau' or 'Transition'
        self.timestamp = timestamp
        self.fs = fs
        self.Npts = int(minutes * 60 * fs)

        #--- Events data ---
        periods = os.path.join(
                os.path.dirname(self.path),
                f"periods-{self.name}.json")
        try:
            with open(periods) as data:
                self.periods = json.load(data)
        except:
            self.periods = {}
            print(f"No events metadata found: {periods}\n"
                + f"Run scripts-infusion/extract_timestamps.py")

        #--- Results ---
        results = os.path.join(
                os.path.dirname(self.path),
                f"results-{self.name}.json")
        try:
            with open(results) as data:
                self.results = json.load(data)
        except:
            self.results = {}
            print(f"No results metadata found: {results}\n"
                + f"Run scripts-infusion/extract_results.py")

    def __iter__(self):
        """ 
        Yield `(key, icp)` pairs of file keys and ICP signal slices.

        The `icp` tensor is of shape `dset.Npts`. 
        """
        #--- Yield ICP signal slices 
        keys, start = self.items_start()
        for k, i0 in zip(keys, start):
            file = self.get(k)
            try:
                icp = file.icp(i0, self.Npts)
                if icp.shape[0] != self.Npts:
                    raise RuntimeError("not enough points")
                yield k, icp
            except Exception as e:
                pass
            file.close()

    def items_start(self):
        """
        Return a pair `(keys, start)` of iterables. 

        Every key is kept if only if the timestamp was found 
        and there remains more than `dset.Npts` time points after
        the associated `start` index. 

        Override this method if working with different hdf5 metadata formats. 
        """
        #--- Filter files with timestamp if any 
        ts = self.timestamp
        if ts is not None:
            keys = self.filter(lambda f: self.periods[f.key][ts])
            evts = self.periods
            start = [int(self.fs * (evts[k][ts][0] - evts[k]["start"])) for k in keys]
        else: 
            keys = self.ls()
            start = [0] * len(keys)
        return keys, start

    def __len__(self):
        return len(self.items_start()[1])

    def ls (self):
        """ List exam files in the dataset. """
        return os.listdir(self.path)

    #--- File opening ---

    def file (self, path):
        """ Get file with exact name. """
        return File(os.path.join(self.path, path), self)

    def get (self, key):
        """ Return first file instance whose name matches pattern. """
        return self.file(self.find(key))

    def getAll (self, *patterns):
        """ Return all file instances whose names match one pattern. """
        if len(patterns) == 0:
            patterns = [r'']
        return [self.file(f) for f in self.findAll(*patterns)]

    #--- Filters and maps ---

    def filters (self, reader):
        """ Map exam files not raising exceptions. """
        out = []
        for name in self.ls():
            f = self.file(name)
            try:
                out += [reader(f)]
            except:
                pass
            f.close()
        return out

    def filter (self, test):
        """ Filter exam filenames that do not raise exceptions. """
        def key (file):
            test(file)
            return file.key
        return self.filters(key)

    def map (self, f):
        """ Return a filtered map of files through f. """
        def reader (file):
            return file.key, f(file)
        return {k: vk for k, vk in self.filters(reader)}

    #--- Filename queries ---

    def find (self, pattern):
        """ Get first file name matching pattern. """
        files = self.ls()
        if isinstance(pattern, int):
            return files[pattern % len(files)]
        for f in files:
            if re.match(pattern, f):
                return f

    def findAll (self, *patterns):
        """ Get all file names matching pattern. """
        if not len(patterns):
            patterns = [r'']
        out = []
        for p in patterns:
            out += [f for f in self.ls() if re.match(p, f)]
        return out

    #--- Show ---

    def __repr__(self):
        return f"Dataset {self.name} "\
             + f"({len(self.ls())} infusion files)"
