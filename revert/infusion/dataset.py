import os
import re
import json

from .file import File

PREFIX = (os.environ["INFUSION_DATASETS"]
            if "INFUSION_DATASETS" in os.environ
            else os.getcwd())

class Dataset:

    def __init__(self, path='2016'):
        """ Provide a path to the dataset. """
        self.name = os.path.basename(path)
        self.path = (path if path[0] == "/"
                        else os.path.join(PREFIX, path))
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
            print(f"No events metadata found: {results}\n"
                + f"Run scripts-infusion/extract_results.py")

    def __iter__(self):
        for f in self.ls():
            file = self.get(f)
            yield file
            file.close()

    def ls (self):
        """ List exam files in the dataset. """
        return os.listdir(self.path)
        p = path if path[0] == "/" \
                 else os.path.join(os.path.dirname(__file__), path)

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
