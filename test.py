import time
from tqdm import tqdm, trange

for i in trange(100, position=0):
    for j in trange(20, position=1):
        time.sleep(0.01)
