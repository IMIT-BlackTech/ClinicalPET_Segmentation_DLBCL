
from info.med import radiomics_features
from info.me import io
import nibabel as nib
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
import os



root = r'/home/nimo/nas_nfs/DLBCL/predict_outDir/'
dirs = io.leaf_folders(data=root)
for i, dir in enumerate(dirs):
    print(i, dir)

crit = (lambda x: '/CTres' in x or 'epoch2000' in x or '/SUV' in x)
load_array = (lambda x: nib.load(x).get_fdata().transpose((2, 0, 1)))  # .transpose((2, 0, 1))
spacing = (lambda x: np.linalg.norm(nib.load(x).affine[:3, :3], ord=2, axis=0)[::-1])


def gen():
    for i, f in enumerate(io.leaf_folders(data=root)):
        file_num = os.listdir(f)
        if len(file_num) < 5:
            print(f'{f} has only {len(file_num)} files, skipping...')
            continue
        _m = [_ for _ in io.search_from_root(data=f, search_condition=crit)]
        msk, suv, ct = [load_array(_) for _ in _m]
        sp = spacing(_m[2])
        print(f'now processing {i} {f}...')
        yield f, suv, msk, sp


print(gen())

df: DataFrame = radiomics_features(data=gen())
df.to_csv('SUV_radiomics.csv')
