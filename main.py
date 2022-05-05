import kuramoto
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


if __name__=="__main__":
    dt = 0.05
    tmax = 20
    
    obj = kuramoto.SpatialKuramoto(100, 20, w_a=1, w_b=2)
    obj.run(tmax=tmax, dt=dt)
    
    with open("data/phase.pkl", "wb") as fid:
        pkl.dump(obj.phase_all, fid)
    
    t_range = np.linspace(0, len(obj.phase_all)-1, 11, dtype=int)
    for t in t_range:
        im = np.sin(obj.phase_all[t].reshape(100,100))
        
        plt.figure(dpi=150, figsize=(5,5))
        plt.imshow(im, cmap="RdBu", vmin=-1, vmax=1)
        plt.title("n = %d"%(t), fontsize=14)
        plt.axis("off")
        plt.savefig("./test/step%05d.png"%(t))
        plt.close()
