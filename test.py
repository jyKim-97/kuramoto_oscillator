import kuramoto
import matplotlib.pyplot as plt
import numpy as np


if __name__=="__main__":
    obj = kuramoto.SpatialKuramoto(100, 20, w_a=1, w_b=2)
    obj.run(tmax=20, dt=0.05)
	
    im = np.sin(obj.phase_all[-1].reshape(100,100))
    plt.figure(dpi=150, figsize=(5,5))
    plt.imshow(im, cmap="RdBu", vmin=-1, vmax=1)
    plt.axis("off")
    plt.savefig("./test.png")
    plt.close()
