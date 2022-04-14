def shift_all (stdev):
    def run_shift(x):
        N = len(x)
        Nc = x.shape[1]
        #generate and convert to tensor
        idx = torch.arange(32).repeat(Nc*N).view([Nc*N, 32])
        # repeat for all data        
        y = torch.randn([N, Nc]) * stdev
        Npts = x.shape[-1]

        y = mod(y, 1)
        y = (y - y.mean([1])[:,None])

        y_index = (y * (Npts / 2)).flatten().long() 

        idx = (idx + y_index[:,None]) % Npts
        idx = (torch.arange(Nc*N)[:,None] * 32 + idx).flatten()
        x_prime = x.flatten()[idx].view([N, Nc, Npts])
        return x_prime, y
    
    return run_shift
