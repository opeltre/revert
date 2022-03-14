import re
import torch
import h5py
import xml.etree.ElementTree as xml
import datetime
import math

def unixTime (string): 
    fmt = "%d/%m/%Y %H:%M:%S"
    dt = datetime.datetime.strptime(string, fmt)
    return dt.timestamp()


class File (h5py.File): 

    def __init__(self, path, db=None):
        """ 
        Create a file instance from exact path.  

        N.B: more easily done with `db.get(pattern)`  
        """ 
        match = re.match(r'^.*/(.*)\.hdf5', path)
        self.key = match.group(1) if match else "none"
        self.db  = db
        super().__init__(path, 'r')

    #--- ICP signal --- 

    def icp (self, N1=None, N2=None, **kwargs):
        """ 
        Return a slice of ICP signal from O to N1 or N1 to N1 + N2. 

        Usage :
        -------
            f.icp()
                Load full ICP signal

            f.icp(N1)    
                Load N pts of ICP signal from start
            
            f.icp(N0, N1)
                Load N1 pts of ICP signal from point N0.

            f.icp(N1, infusion=-N0)
                Load N1 pts of ICP signal from infusion start + (-N0). 
        """
        if N1 == None:
            return torch.tensor(self['waves']['icp'])
        if "infusion" in kwargs:
            N2 = N1
            N1 = int(self.infusion_start() * self.fs())\
               + int(kwargs["infusion"])
        elif N2 == None:
            N1, N2 = 0, N1
        array = self['waves']['icp'][N1:N1+N2]
        return torch.tensor(array)

    def fs (self):
        """ Return sampling frequency. """
        return self.index()['frequency']
    
    #--- Patient Information --- 

    def info (self):
        """ Return patient info as dict. """
        pairs = list(self.get('patient.info'))
        dec   = lambda bs : bs.decode("utf-8")
        return {dec(p[0]): dec(p[1]) for p in pairs}

    def age (self):
        """ Parse age from patient information. """
        return int(re.match(r'\d*', self.info()["age"]).group(0))

    #--- Events --- 

    def nblocks (self):
        """ Return number of blocks in ICP signal. """
        return len(self.get('waves/icp.index'))

    def index (self, i=0):
        """ Return block information as dict. """
        return self.table('waves/icp.index')[i]

    def start (self, i=0):
        """ Return start of block UNIX time (ms). """
        return self.index(i)['starttime']

    def nevents (self): 
        """ Return number of events in annotations. """
        return len(self.get("annotations/events"))

    def events (self): 
        """ Return information of event i. """
        return self.table("annotations/events")

    #--- Datatypes --- 

    def bytestring (self, key):
        """ Extract bytestring at file.get(key). """
        return list(self.get(key))[0]

    def xml (self, key):
        """ Return XML instance from bytestring at file.get(key). """
        return xml.fromstring(self.bytestring(key))

    def table (self, key): 
        """ Extract dict from table row i at file.get(key). """
        obj = self.get(key)
        name = obj.dtype.names
        return [
            {name[i]: vi for i, vi in enumerate(val)}\
            for val in obj]

    def unix (self, t):
        return unixTime(t)

    def datetime (self, t):
        pass

    #--- Timestamps ---
    
    def infusion_start(self, src='icmtests'):
        """
        Return start of infusion in seconds from start of recording.

        Look for file.key in file.db.periods when it exists.
        Otherwise try to parse chosen XML source.

        Arguments:  
        ----------
            - src : 'icmtests' | 'icmevents' | 'annotations' 
        """
        #--- Look for json metadata --- 
        db = self.db
        if db and self.key in db.periods:
            periods = db.periods[self.key]
            return periods['infusion'][0] - periods['start']
        #--- Otherwise parse chosen xml source ---
        return self.interval('Infusion', src)[0]\
             - self.start()

    def intervals(self, src="icmtests", fmt="unix"):
        """ Return a dictionnary of the form {evtname : [t1, t2]}. """
        #--- Event source ---
        if src == "annotations":
            evts = self.events()
        elif src == "icmevents":
            tree = self.xml("aux/ICM+/icmevents")
            evts = tree.find("Events").findall("Event")
        else: 
            tree = self.analysis()
            evts = tree.find("Selections").findall("Selection")
    
        #--- Fix timezone incoherence ---
        def tzone (t):
            begin = self.start()
            h = math.ceil((begin - t) / 3600)
            return t - begin + h * 3600

        #--- [Start, End]
        def timestamp (evt):
            if src == "annotations":
                t0 = evt['starttime'] // 10**6
                t0 = tzone(t0)
                return [t0, t0 + evt['duration'] // 10**6]
            attr = evt.attrib if src == "icmevents"\
                   else evt.find("TimePeriod").attrib
            ts  = [attr["StartTime"], attr["EndTime"]]
            return [tzone(unixTime(t)) for t in ts] \
                   if fmt == "unix" else ts
    
        #--- Event name ---
        def key(evt):
            return (evt['eventname'].decode('utf8') if src == 'annotations'\
                   else evt.attrib['Name'])

        return {key(e): timestamp(e) for e in evts}
        
    def interval(self, key, src="icmtests", fmt="unix"):
        ts = self.intervals(src, fmt=fmt)
        ks = [k for k in ts if re.search(key, k)]
        return ts[ks[0]]

    def start(self, src="index", fmt="unix"):
        return self.index()['starttime'] // 10**6

    def total_interval (self, fmt="unix"):
        log = self.get("aux/ICM+/DTAFilesLog")
        b = list(log)[9]
        s = b.decode("utf8")
        reg = r'\d\d/\d\d/\d{4}\s\d\d:\d\d:\d\d'
        ts = re.findall(reg, s)
        return ts if fmt != "unix" else [unixTime(t) for t in ts]

    #--- Regression Parameters ---

    def analysis (self):
        return self.xml('aux/ICM+/icmtests')\
                .find("AnalysisInstances")  \
                .find("SingleAnalysis")

    def results (self):
        params = self.analysis().find("Results").findall("Parameter")
        attrs  = [e.attrib for e in params]
        return {a["Name"]: float(a["Value"]) for a in attrs}

    def values (self, *patterns):
        if not(len(patterns)):
            patterns = ['']
        res, out = self.results(), {} 
        for p in patterns:
            out[p] = {k: rk for k, rk in res.items() \
                                      if re.match(p,k)}
        return out

    def value (self, pattern):
        return self.values(pattern)[pattern]

    #--- Show --- 

    def __repr__(self): 
        def gstr(group): 
            s = ''
            for k in group.keys():
                if isinstance(group[k], h5py.Dataset): 
                    s += f'\n{k}: {group[k].shape}'
                else:
                    sk = gstr(group[k]).replace('\n', '\n. ')
                    s += f'\n{k}: {sk}'
            return s 
        return '<Hdf5>' + gstr(self).replace('\n', '\n  ')
    
