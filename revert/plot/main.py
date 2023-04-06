import matplotlib.pyplot as plt

show = plt.show

def images(W, size=(8, 4)):
    if W.dim() == 3:
        n = W.shape[0]
        h, w = 1, n
    elif W.dim() == 4:
        h, w = W.shape[0:2]
        n = h * w
        W = W.flatten(end_dim=1)
        print(h, w, W.shape)
    plt.figure(figsize=size)
    for i in range(n):
        plt.subplot(h, w, i+1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.imshow(W[i])