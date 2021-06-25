import h5py
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

filename = "test.h5"

def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


'''with h5py.File(filename, 'r') as f:
    for dset in traverse_datasets(f):
        print('Path:', dset)
        print('Shape:', f[dset].shape)
        print('Data type:', f[dset].dtype)
        if dset == '/table/table':
            #print(f[dset][:])
            print(f[dset][:][0][0])
            print(f[dset][:][0][1])
            print(f[dset][:][0][2])
        #print(f[dset][:])'''

with pd.HDFStore(filename, mode = 'r') as store:
    df = store['table']


def plot_energies(jet_series):
    energies = [jet_series[f'E_{i}'] for i in range(200)]
    sns.distplot(energies)


print(df)
sns.displot([df.iloc[i]["E_0"] for i in range(10000)])
plt.show()
