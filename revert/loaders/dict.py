class Dict (dict): 
    def fmap(self, f):
        return Dict({k: f(v) for v , k in self})
    
    def fmap_(self, f):
        for v, k in self:
            self[k] = f(v)
        return self

    def map(self, f): 
        return Dict({k: f(v, k) for v, k in self})

    def map_(self, f):
        for v, k in self:
            self[k] = f(v, k)
        return self

    def filter(self, f):
        out = Dict({})
        for v, k in self:
            if f(v, k):
                out[k] = v
        return out

    def filter_(self, f): 
        keys = []
        for v, k in self: 
            if not f(v, k):
                keys += [k]
        for k in keys:
            self.pop(k)
        return self

    def pluck(self, *keys): 
        return Dict({k : self[k] for k in keys})
   
    def reduce(self, f, acc=None):
        if acc == None:
             return Dict(self.copy()).reduce_(f)
        return self.reduce_(f, acc)

    def reduce_(self, f, acc=None):
        if acc == None:
            (k, acc) = self.popitem()
        for v, k in self:
            acc = f(acc, v, k)
        return acc
        
    def __iter__(self): 
        return ((self[k], k) for k in super().__iter__())

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __dir__(self):
        return super().__dir__() + [str(k) for k in self.keys()]

    def __repr__(self): 
        s = "{\n"
        for v, k in self: 
            sv = v.__repr__().replace("\n", "\n    ")
            s += f"    {k}\t: {sv},\n"
        return s[:-2]+ "\n}"
