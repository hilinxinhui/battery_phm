import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt

import numpy as np

# config = {
#     "font.family": "serif",
#     "font.size": 20,
#     "mathtext.fontset": "stix",
#     "font.serif": ["SimSun"],
# }
# rcParams.update(config)

def get_mpl_info():
    print(mpl.matplotlib_fname()) # matplotlibrc文件
    print(mpl.get_cachedir()) # matplotlib缓存目录

plt.rcParams['font.serif'] = ['STSong']
    
get_mpl_info()

index = np.arange(-100, 100)
data = np.array([i ** 2 for i in index])
plt.title("标题，title")
plt.plot(index, data)
plt.xlim((-100, 100))
plt.ylim((-10, 10000))
plt.xlabel("横坐标（index）")
plt.ylabel("纵坐标（data）")
plt.savefig("./use_STsong.jpg", dpi=1000, bbox_inches="tight")
    