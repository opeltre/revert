from matplotlib import pyplot as plt

def style(name='seaborn'):
    plt.style.use('seaborn')

def colors():
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return c

#--- Multiple plots 

def grid (traces, shape, title=None, titles=None, c=None, lw=1):
    c = colors() if not c else c
    w, h = shape[:2]
    sw, sh = shape[2:] if len(shape) > 2 else (4, 4)
    fig = plt.figure(figsize=(w * sw, h * sh))
    if title:
        plt.suptitle(title)
    for i, t in enumerate(traces):
        ax = plt.subplot(h, w, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.plot(t, color=c[i % len(c)], lw=lw)
        if titles:
            plt.title(titles[i], {'fontsize': 8}, y=0.93)
    return fig

#--- KMeans plots 

def cluster_grid (km, x, y, P=12, avg=False, title="Cluster grid"):
    ids = km.nearest(P, y)
    nearest = [x.index_select(0, js) for js in ids]
    traces  = [xs.T if not avg else xs.mean([0]) for xs in nearest]
    masses  = km.counts(y) / y.shape[0]
    vars    = km.vars(y) / y.shape[-1]
    titles  = [f'[{i}]    mass {100 * mi:.1f}% - var {vars[i]:.2f}'\
                    for i, mi in enumerate(masses)]
    fig = grid(traces, (8, 8), title=title, titles=titles)
    return fig

def cluster(xi, i='n', c='blue'):
    plt.plot(xi[:64].T, color=c, lw=.5)
    plt.title(f'cluster {i}')
    plt.show()

#--- Full ICP recordings

def infusion(icp, events, fs=100, size=(20, 10)):
    # full ICP signal
    fig = plt.figure(figsize=size)
    time = [float(i) / fs for i in range(icp.shape[0])]
    plt.plot(time, icp, color="orange")
    bnd = icp.min(), icp.max()
    # event markers
    def plotEvent(name, color):
        if not name in events: return None
        xs = events[name]
        for i in (0, 1): plt.plot([xs[i]] * 2, bnd, color=color)
    names  = ["Baseline", "Infusion", "Plateau"]
    colors = ["blue", "red", "green"]
    for n, c in zip(names, colors): plotEvent(n, c)
    return fig
