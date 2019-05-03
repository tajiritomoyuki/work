#coding:utf-8
import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing as mp
import h5py

dirtype = "step3"
h5dir = "/pike/pipeline/%s" % dirtype
dstdir = "/home/tajiri/tess/nn/data"

def load(h5path):
    with h5py.File(h5path, "r") as f:
        flux = np.array(f["LC"]["SAP_FLUX"])
        quality = np.array(f["LC"]["QUALITY"])
        mid_val = np.nanmedian(flux)
        if mid_val == 0:
            lc = np.zeros_like(flux)
        else:
            lc = flux / mid_val
        lc_interp = np.copy(lc)
        x = lambda z: z.nonzero()[0]
        lc_interp = [quality] = np.interp(x(quality), x(~quality), lc_interp[~quality])
    return lc_interp, os.path.basename(h5path)

def main():
    for sector in range(1, 8)
        h5list = glob.glob(os.path.join(h5dir, "*_%s_?_?.h5" % sector))
        lc_list = []
        path_list = []
        #読み込み
        with mp.Pool(mp.cpu_count()) as p:
            for lc, path in p.map(load, tqdm(h5list)):
                lc_list.append(lc)
                path_list.append(path)
        lc_array = np.vstack(tuple(lc_list))
        path_array = np.hstack(tuple(path_list))
        #書き出し
        dstpath = os.path.join(dstdir, "data_%s.h5" % sector)
        with h5py.File(dstpath, "w") as f:
            f.create_dataset("lc", data=lc_array)
            f.create_dataset("path", data=path_array)

if __name__ == '__main__':
    main()
