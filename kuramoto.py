import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl
from tqdm import tqdm
import cv2


class SpatialKuramoto:
    """
    Simulate Kuramoto oscillator
    ============================
    Input:

    N: The number of the oscillators
    K: coupling strength
    natural_freq_distrib: probability distribution type of the natural frequency
        - const: delta function distribution, need "w" as arguments which determines natural freq.
        - uniform: uniform distribution, need "w_a", "w_b" as arguments which determine the range for prob.
        - gaussian: gaussian distribution, need "w_mu", "w_sgm" as arguments which determine mean & standard deviation of prob.
    
    network: type of the network
        - full: fully connected network (all-to-all)
        - sptial: connect only nearest neighbor units

    natural_params: arguemnts of natural_freq_distrib. see above.

    ============================
    Example
    obj = ko.SpatialKuramoto(10000, 1, "gaussian", w_mu=3, w_sgm=1, network="spatial")
    obj.run(tmax=10, dt=0.01)
    obj.save(<file_name_to_save_phase_data_to_time>)
    """

    def __init__(self, N, K, natural_freq_distrib="uniform", network="spatial", **natural_params):
        self.K = K
        self.num_osc = N
        self.L = int(np.sqrt(N))
        self._set_simulation(natural_freq_distrib, network, natural_params)
        
    def run(self, tmax=100, dt=0.01, save_name=None):
        # tmax: maximum simulation time
        # dt: time step of the simulation
        self.dt = dt
        self.tmax = tmax
        self.ts = np.arange(0, tmax, dt)
        self.num_itr = int(tmax/dt)
        self.phase = np.random.random(self.num_osc)*2*np.pi
        self.phase_all = np.zeros([self.num_itr, self.num_osc])
        
        for n in tqdm(range(self.num_itr)):
            self.update()
            self.phase_all[n] = self.phase
        
    def f(self, p):
        dp = np.zeros(self.num_osc)
        for n in range(self.num_osc):
            phs_post = p[self.ntk[n]]
            dp[n] = np.average(np.sin(phs_post - p[n]))
        return self.w + dp * self.K
        
    def update(self):
        k1 = self.f(self.phase) * self.dt
        k2 = self.f(self.phase+k1/2) * self.dt
        k3 = self.f(self.phase+k2/2) * self.dt
        k4 = self.f(self.phase+k3) * self.dt
        dp = (k1 + 2*k2 + 2*k3 + k4)/6
        self.phase = self.phase + dp
    
    def _set_simulation(self, natural_freq_distrib, network_type, natural_params):
        self.phase = np.zeros(self.num_osc)
        if network_type == "spatial":
            self._gen_network_spatial()
        elif network_type == "full":
            self._gen_network_full()
        else:
            print("Selected wrong network type")
        
        if natural_freq_distrib == "uniform":
            self._set_natural_freq_uniform(natural_params)
        elif natural_freq_distrib == "gaussian":
            self._set_natural_freq_gaussian(natural_params)
        elif natural_freq_distrib == "const":
            self._set_natural_freq_const(natural_params)
        else:
            print("Selected wrong prob distribution")
        
    def _gen_network_spatial(self):
        # periodic boundary condition
        self.ntk = []
        # DIR = (1, -1, self.L, -self.L)
        for i in range(self.num_osc):
            DIR = [1, -1, self.L, -self.L]
            
            self.ntk.append([])
            if i % self.L == 0:
                DIR[1] += self.L
            elif i % self.L == self.L-1:
                DIR[0] -= self.L
            if i // self.L == 0:
                DIR[3] += self.num_osc
            elif i // self.L == self.L-1:
                DIR[2] -= self.num_osc
            
            for d in DIR:
                n_post = i + d
                self.ntk[-1].append(n_post)
    
    def _gen_network_full(self):
        self.ntk = []
        for i in range(self.num_osc):
            tmp = [j for j in range(self.num_osc) if j != i]
            self.ntk.append(tmp)
    
    def _set_natural_freq_gaussian(self, natural_params):
        self._check_key_in(natural_params, "w_mu", "w_sgm")
        self.w = np.random.random(self.num_osc) * natural_params["w_sgm"] + natural_params["w_mu"]
    
    def _set_natural_freq_uniform(self, natural_params):
        self._check_key_in(natural_params, "w_a", "w_b")
        l = natural_params["w_b"] - natural_params["w_a"]
        self.w = np.random.randn(self.num_osc) * l + natural_params["w_a"]
    
    def _set_natural_freq_const(self, natural_params):
        self._check_key_in(natural_params, "w")
        self.w = np.ones(self.num_osc) * natural_params["w"]
        
    def _check_key_in(self, dict_param, *key_names):
        keys_dict = dict_param.keys()
        for k in key_names:
            if k not in keys_dict:
                print(f"{k} is not given as the parameters")
        
    def save(self, save_name):
        print("print data to %s"%(save_name))
        with open(save_name, "wb") as fid:
            pkl.dump(fid, self.phase_all)
        
    def load(self, fname):
        print("load data %s"%(fname))
        with open(fname, "rb") as fid:
            self.phase_all = pkl.load(fid)
                

class VideoMaker:
    """
    Make video of the simulation data 
    ============================
    Input:
    objKuramoto: SpatialKuramoto object of "spatial" network type after execute "run" method
    ============================
    Example
    video = VideoMaker(obj)
    video.export(<video_name>, frame_itv=1, fps=30)
    """
    def __init__(self, objKuramoto):
        self.obj = objKuramoto
        self.ims = np.sin(reshape(self.obj.phase_all))
    
    def export(self, out_name=None, dpi=120, figsize=(4,4), frame_itv=1, fps=30):
        # frame_itv: simulation step interval of the video
        # fps: video fps
        self.draw_init(dpi, figsize)
        self.init_video(out_name, fps)
        
        num_frames = self.ims.shape[0]//frame_itv
        for n in tqdm(range(num_frames)):
            self.draw(n*frame_itv)
            im = get_frame(self.fig)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            self.out.write(im)
        
        self.out.release()
    
    def init_video(self, out_name, fps):
        if out_name is None:
            date = time.localtime()
            out_name = "%02d%02d%02d.mp4"%(date.tm_hour, date.tm_min, date.tm_sec)
            
        frame = get_frame(self.fig)
        frame_size = frame.shape[:-1]
        
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter(out_name, fourcc, fps, frame_size)
    
    def draw_init(self, dpi, figsize):
        self.fig = plt.figure(dpi=dpi, figsize=figsize)
        self.im_map = plt.imshow(self.ims[0], cmap='RdBu', vmax=1, vmin=-1)
        plt.axis('off')
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def draw(self, n_frame):
        self.im_map.set_array(self.ims[n_frame])
        self.fig.canvas.draw()
    
    
def get_frame(fig):
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                
def reshape_single(phase):
    L = int(np.sqrt(phase.shape[0]))
    return phase.reshape(L, L)


def reshape(phase):
    N = phase.shape[0]
    L = int(np.sqrt(phase.shape[1]))
    return phase.reshape(N, L, L)


def get_r(phase_all):
    return np.abs(np.average(np.exp(1j*phase_all), axis=1))


if __name__ == "__main__":
    obj = SpatialKuramoto(100, 20, w_a=1, w_b=3)
    obj.run(tmax=20, dt=0.05)
    Video = VideoMaker(obj)
    Video.export("test.mp4")
