#!/usr/bin/env python

import sys, getopt
import glob, os
import numpy as np
import spams
import multiprocessing
import glob


hash_size = 30
#n = 1135
#n = 10
cpu = 30
clusters_nb = 1
thres = 0
wd = ""

name = ""
name2 = ""
prefix = ""
suffix = ""


if len(sys.argv) > 1:
	name = sys.argv[1]
if len(sys.argv) > 2:
	name2 = sys.argv[2]
if len(sys.argv) > 3:
	prefix = sys.argv[3]	

########### Load matrix


global_nzi = np.zeros((2**30), dtype = np.bool)
values = {}
nzis = {}
n0 = 0


files = glob.glob(prefix + '*.count.hash')
files = set(files)

n = len(files)


for k in range(n0, n + n0):
	print("read nzi " + str(k))
	nzis[k] = np.fromfile(prefix + str(k) + ".count.hash.nzi", dtype='uint64')
	print(nzis[k])
	global_nzi[nzis[k]] = True


print("save data ")

np.save(wd + "matrices/global_nzi" + name + ".npy", global_nzi)
nz = np.sum(global_nzi)
inverted_index = np.cumsum(global_nzi) - 1


norm = np.zeros((2**hash_size), dtype = np.float64)
mean = np.zeros((2**hash_size), dtype = np.float64)


for k in range(n0, n + n0):
	print("read hash " + str(k))
	values[k] = np.fromfile(prefix + str(k) + ".count.hash", dtype='uint16')
	norm[nzis[k]] += values[k].astype(np.float64) ** 2
	mean[nzis[k]] += values[k].astype(np.float64)


mean = mean/n
norm = np.sqrt(norm)
norm[norm == 0] = 1


div = int(nz / (3e7))
chunk_size = int(nz/div)
global_inds = np.arange(2**hash_size)[global_nzi]


param = { 'K' : 100, 
          'lambda1' : 0.1, 
          'lambda2' : 0.1, 
          'posAlpha' : True, 'numThreads' : cpu, 'batchsize' : 10000,
          'iter' : 10000, 'posD' : False}


D = None
sections = np.arange(div)

for k in sections[:-1]:
	print("compute matrix")
	sup = min(nz, chunk_size * (k + 1))

	inds = global_inds[chunk_size * k : sup]
	size_of_matrix = sup - chunk_size * k
	print(size_of_matrix)

	abundance_matrix = np.zeros((n, size_of_matrix), dtype = np.float32)

	for i in range(n0, n + n0):		
		chunk = np.zeros(size_of_matrix)
		inds_of_values = (nzis[i] >= global_inds[chunk_size * k]) * (nzis[i] < global_inds[sup - 1])
		inds_absolute = nzis[i][inds_of_values]
		chunk[inverted_index[inds_absolute] - inverted_index[global_inds[chunk_size * k]]] = values[i][inds_of_values]
		abundance_matrix[i - n0, :] = chunk/norm[inds]


	print("matrix computed")
	print(abundance_matrix)
	
	print(D)
	param['D'] = D

	print(k)
	X = np.asfortranarray( abundance_matrix )
	
	print(X)
	D = spams.trainDL(X, **param)


np.save(wd + "matrices/D" + name + name2, D)


def _extract_lasso_param(f_param):
    lst = [ 'L','lambda1','lambda2','mode','pos', 'ols','numThreads','length_path','verbose','cholesky']
    l_param = {'return_reg_path' : False}
    if 'posAlpha' in f_param:
    	l_param['pos'] = f_param['posAlpha']
    for x in lst:
        if x in f_param:
            l_param[x] = f_param[x]
    return l_param

lparam = _extract_lasso_param(param)
print(lparam)
lparam['numThreads'] = 1

chunk_size = 2**17
iter_nb = int(nz/chunk_size + 1)
alpha = np.memmap(wd + 'matrices/kmer_clusters' + name + name2 , dtype='int16', mode='w+', shape=(5, 2**hash_size), order = 'F')


nz_code = 0


data = np.zeros((n * nz), dtype = np.float32)
indices = np.zeros((n * nz), dtype = np.int16)
indptr = np.zeros(nz + 1, dtype = np.int64)

def write_part(k):
	global nz_code
	print("start " + str(k))
	sup = min(nz - 1, chunk_size * (k + 1))
	
	print("compute matrix")

	inds = global_inds[chunk_size * k : sup]
	size_of_matrix = sup - chunk_size * k
	print(size_of_matrix)

	abundance_matrix = np.zeros((n, size_of_matrix), dtype = np.float32)

	for i in range(n0, n + n0):
		chunk = np.zeros(size_of_matrix)
		inds_of_values = (nzis[i] >= global_inds[chunk_size * k]) * (nzis[i] < global_inds[sup])
		inds_absolute = nzis[i][inds_of_values]
		chunk[inverted_index[inds_absolute] - inverted_index[global_inds[chunk_size * k]]] = values[i][inds_of_values]
		abundance_matrix[i - n0, :] = chunk/norm[inds]

	print("matrix computed")
	print(abundance_matrix)
	
	x = np.asfortranarray(abundance_matrix)


	a = spams.lasso(x, D = D, **lparam)

	nz_chunk = np.shape(a.data)[0]
	indices[nz_code:(nz_chunk + nz_code)] = a.indices[:]
	indptr[chunk_size * k : (sup + 1)] = a.indptr[:] +  indptr[chunk_size * k]
	data[nz_code:(nz_chunk + nz_code)] = a.data[:]

	print(nz_code, nz_chunk)
	nz_code = nz_code + nz_chunk

	a = a.toarray()

	print(np.sum(a > 0, 0))

	clusters = np.argsort(a, axis = 0)[-clusters_nb:][::-1]

	mask = a[clusters,  np.arange(size_of_matrix)]
	mask = mask > thres

	clusters[~mask] = -1
	alpha[:clusters_nb, inds] = clusters + 1
	alpha.flush()

	print(str(i) + " ok !")
	return 0


for k in range(iter_nb):
	write_part(k)


del alpha
print("save data")


data = data[:nz_code]
indices = indices[:nz_code]

print(data)
print(indptr)
print(indices)


np.save(wd + "matrices/data" + name + name2, data)
np.save(wd + "matrices/indptr" + name + name2, indptr)
np.save(wd + "matrices/indices" + name + name2, indices)


