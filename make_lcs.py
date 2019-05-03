#coding:utf-8
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import multiprocessing as mp
import h5py

rootdir = os.path.abspath(os.path.dirname(__file__))
csvpath = os.path.join(rootdir, "csv", "labeled.csv")
pldir = "/pike/pipeline"
dstdir = os.path.join(rootdir, "data")

def load(path):
    h5path1 = os.path.join(pldir, "step3", path)
    h5path2 = os.path.join(pldir, "TIC3", path)
    if os.path.exists(h5path1):
        h5path = h5path1
    else:
        h5path = h5path2
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
        lc_interp = np.interp(x(quality), x(~quality), lc_interp[~quality])
    return lc_interp, os.path.basename(h5path)

def main():
    df = pd.read_csv(csvpath, names=["path", "label"])
    f_sector = lambda x: int(x.split("_")[2])
    sector_df = df["path"].map(f_sector)
    df["sector"] = sector_df
    for label, sector in product([0, 1], [1, 2, 3, 5]):
        tar_df = df[(df["sector"] == sector) & (df["label"] == label)]
        path_list = tar_df["path"].values
        lc_list = []
        #読み込み
        with mp.Pool(2) as p:
            for lc, path in p.map(load, tqdm(path_list)):
                lc_list.append(lc)
        lc_array = np.vstack(tuple(lc_list))
        #書き出し
        dstpath = os.path.join(dstdir, "%s_%s.npz" % (sector, label))
        np.savez(dstpath, data=lc_array)

if __name__ == '__main__':
    main()
