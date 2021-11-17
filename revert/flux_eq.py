from flux_svd import * 

def str_eq (eq):
    sym =   ['fa '  , 'fv ' , 'fcs ' 
            ,"fa'"  , "fv'" , "fcs'"
            ,'fa"'  , 'fv"' , 'fcs"']
    def num (a):
        s = "+" if a > 0 else "-"
        return f"{s} {abs(float(a)):.3f}"
    terms = [f"{num(ti)} {sym[i]} " for i, ti in enumerate(eq)]
    return    "".join(terms[6:9]) + "\n"\
            + "".join(terms[3:6]) + "\n"\
            + "".join(terms[0:3])

def print_eq (eq): 
    print(str_eq(eq))

CC = 10 
print(f"\n--- Cardiac Cycle: {CC} time units ---\n")

f = flux("d3")
j = jet(2, CC/64)(f)
j = j * (1 / j.norm())

print(f"|f | = {j[0].norm():.3f}")
print(f"|f'| = {j[1].norm():.3f}")
print(f'|f"| = {j[2].norm():.3f}\n')

j = j.reshape((9, j.shape[-1]))

u,s,v = torch.svd(j)
eq = u.t()

for (i, ei) in enumerate(eq):
    print(f"\nEq[{i}]: \t (norm {s[i]:.3f})")
    print_eq(ei)
