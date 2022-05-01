import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2


class SpatialKuramoto:
    def __init__(self, L, K, natural_freq_distrib="uniform", **natural_params):
        self.L = L
        self.K = K
        self.num_osc = L*L
        self._set_simulation(natural_freq_distrib, natural_params)
        
    def run(self, tmax=100, dt=0.01, save_as_file=False, save_name=None):
        
        self.dt = dt
        self.tmax = tmax
        self.num_itr = int(tmax/dt)
        self.save_as_file = save_as_file
        self._init_writer(save_name)
        self.phase = np.random.random(self.num_osc)*np.pi
        
        for n in tqdm(range(self.num_itr)):
            self.update()
            self.write(n)
            
        self._close_writer()
        
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
    
    def write(self, nstep):
        if self.save_as_file:
            np.save(self.fp, self.phase)
        else:
            self.phase_all[nstep] = self.phase
    
    def _init_writer(self, save_name):
        if self.save_as_file:
            if save_name is None:
                date = time.localtime()
                save_name = "%02d%02d%02d"%(date.tm_hour, date.tm_min, date.tm_sec)
            self.fp = fopen(save_name+".npy", "wb")
            self.phase_all = None
        else:
            self.phase_all = np.zeros([self.num_itr, self.num_osc])
        
    def _close_writer(self):
        if self.save_as_file:
            self.fp.close()
    
    def _set_simulation(self, natural_freq_distrib, natural_params):
        self.phase = np.zeros(self.num_osc)
        self._gen_network()
        
        if natural_freq_distrib == "uniform":
            self._set_natural_freq_uniform(natural_params)
        elif natural_freq_distrib == "gaussian":
            self._set_natural_freq_gaussian(natural_params)
        elif natural_freq_distrib == "const":
            self._set_natural_freq_const(natural_params)
        else:
            print("Selected wrong prob distribution")
        
    def _gen_network(self):
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


class VideoMaker:
    def __init__(self, objKuramoto):
        self.obj = objKuramoto
        self.ims = np.sin(reshape(self.obj.phase_all))
    
    def export(self, out_name=None, dpi=120, figsize=(4,4), frame_itv=1, fps=30):
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

        # get frame size
        # self.frame = get_frame(self.fig)
    
    def draw(self, n_frame):
        self.im_map.set_array(self.ims[n_frame])
        self.fig.canvas.draw()
    
    
def get_frame(fig):
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                
def draw_single_im(frame):
    im_map.set_data(np.sin(ims[frame]))
    return im_map,
                
def draw_animation(objKuramoto, out_name=None, dpi=120, figsize=(4,4), frame_itv=1, fps=30, cmap='RdBu'):
    from matplotlib.animation import FuncAnimation, writers
    
    sin_phs = reshape(np.sin(objKuramoto.phase_all))
    
    fig = plt.figure(dpi=dpi, figsize=figsize)
    im_map = plt.imshow(sin_phs[0], cmap=cmap, vmax=1, vmin=-1)
    plt.axis('off')
    
    ani = FuncAnimation(fig, draw_single_im, frames=np.arange(0, ims.shape[0], 1), blit=True)
    writer = writers['ffmpeg'](fps=60)
    
    if out_name is None:
        date = time.localtime()
        save_name = "%02d%02d%02d"%(date.tm_hour, date.tm_min, date.tm_sec)
    print(f"save result as {out_name}.mp4")
    ani.save(f"{out_name}.mp4", writer=writer, dpi=dpi)
    print("Done")
                
                
def reshape_single(phase):
    L = int(np.sqrt(phase.shape[0]))
    return phase.reshape(L, L)


def reshape(phase):
    N = phase.shape[0]
    L = int(np.sqrt(phase.shape[1]))
    return phase.reshape(N, L, L)


if __name__ == "__main__":
    obj = SpatialKuramoto(100, 20, w_a=1, w_b=3)
    obj.run(tmax=20, dt=0.05)
    Video = VideoMaker(obj)
    Video.export("test.mp4")
