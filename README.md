# Kuramoto oscillator simulator
Simulation code for Kuramoto oscillator

| Weak interaction | Strong interaction |  
| --- | --- |
| <img src="https://github.com/jyKim-97/kuramoto_oscillator/blob/master/samples/weak_osc.gif" with="100" height="100" /> | <img src="https://github.com/jyKim-97/kuramoto_oscillator/blob/master/samples/strong_osc.gif" with="500" height="500" /> |


# Run simulation

Code example
```
import Kuramoto as ko

# Run simulation with 10,000 oscillators connecting only nearest neighbors with coupling strength 10
obj = ko.SpatialKuramoto(10000, 10, "gaussian", w_mu=2, w_sgm=1, network="spatial")

# maximal time = 10 with time interval = 0.01
obj.run(tmax=10, dt=0.01)

# Export result as a video every 5 frame
video = ko.VideoMaker(obj)
video.export(out_name="sample.mp4", frame_itv=5, fps=30)
```



