import numpy as np
from numpy.lib import recfunctions as rfn
import tonic 
from dv import AedatFile
import skimage.measure
import matplotlib.pyplot as plt


path = '22_october_box/6.aedat4'
with AedatFile(path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])

x = events['x']
y = events['y']
p = events['polarity']
t = events['timestamp']
data_time = t[0]-t[-1]
data = np.stack((x,y,t,p))
data = np.array(data)
data = np.transpose(data)
dt = np.dtype([("x","int"),("y","int"),("t","int"),("p","int")])
data = rfn.unstructured_to_structured(data, dt)
transform = tonic.transforms.ToFrame(
    sensor_size=(346, 260, 2),
    time_window=50000,
)
frames = transform(data)

animation = tonic.utils.plot_animation(frames=frames)