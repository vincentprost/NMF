#!/usr/bin/env python
from __future__ import division


import sys, getopt
import glob, os
import gzip, re

import numpy as np
from scipy.spatial import distance
import random
from collections import defaultdict
import scipy.sparse as sp



def rank1(C):
	l,c = np.shape(C)
	I = 0
	u = np.ones(l)
	w = np.ones(c)
	
	diff = np.inf
	while diff > 0:
		w = np.transpose(u).dot(C)
		w = w/np.linalg.norm(w)
		u = C.dot(w)
		u = u/np.linalg.norm(u)
		I_ = np.transpose(u).dot(C).dot(w)
		diff = np.abs(I - I_)
		I = I_
	return u, I

def svd(X, dim = -1):

	C = X.dot(np.transpose(X))

	S = np.shape(C)[0]

	if dim == -1:
		dim = np.shape(C)[0]

	U = np.zeros((S,dim))
	I = np.zeros(dim)
	Id = np.eye(S,dim)

	R = C

	for k in range(0, dim):

		u, i = rank1(R)
		
		Z = np.zeros((S, dim))
		Z[:,k] = u

		R = R - i * Z.dot(Z.transpose())

		U[:,k] = u
		I[k] = i
	return I, U


def nmf(X):

	C = X.dot(np.transpose(X))
	S = np.shape(C)[0]

	U = np.zeros((S,S))
	I = np.zeros(S)
	Id = np.eye(S)

	for k in range(0, S):

		u, i = rank1(C)
		U[:,k] = u
		I[k] = i

		T = Id - U.dot(np.linalg.pinv(U))
		R = T.dot(X)
		R[R < 0] = 0
		C = R.dot(R.transpose())

	return I, U


def sparseness(v):
	n = np.shape(v)[0]
	L1 = np.linalg.norm(v, ord=1)
	L2 = np.linalg.norm(v)
	return (np.sqrt(n) - L1/L2)/(np.sqrt(n) - 1)


def nn_sparse_constrained_proj(x, L1, L2):

	n = np.shape(x)[0]
	I = np.ones(n)
	Z = np.ones(n)


	s = x - (L1 - np.sum(x)) / n * I

	count = 0
	while True:
		count += 1
		if np.sum(Z) == 0:
			m = np.zeros(n)
		else:
			m = Z * L1 / np.sum(Z) 

		coeff = [np.linalg.norm(s - m)**2,  2 * np.sum(m *(s - m)) , np.linalg.norm(m)**2 - L2**2]
	

		alpha = np.roots(coeff)
		print(coeff)
		print(alpha)
		
		alpha = alpha[0]
		s = m + alpha * (s - m)


		if np.all(s >= 0):
			print("projection ok")
			print(count)
			break

		
		Z[s < 0] = 0
		mask = Z == 1
		s[s < 0] = 0
		c = 0
		if np.sum(Z) > 0:
			c = (np.sum(s) - L1)/np.sum(Z)
		s[mask] =  s[mask] - c * I[mask]

	return s




def lee_H(V, W, H):
	return H * W.transpose().dot(V) / (W.transpose().dot(W).dot(H))
	

def lee_W(V, W, H):
	return W * V.dot(H.transpose()) / (W.dot(H).dot(H.transpose())) 

def nmf_lee(X, dim, spars = 0.3):

	step = 5
	m, n = np.shape(X)

	W = np.random.rand(m, dim)
	H = np.random.rand(dim, n)
	error = np.inf

	while True:
		W = lee_W(X, W, H)
		H = H - step * W.transpose().dot(W.dot(H) - X)

		for k in range(0, dim):
			x = H[k,:]
			L2 = np.linalg.norm(x)
			L1 = L2 * (np.sqrt(n) - spars * (np.sqrt(n) - 1))

			
			print("former sparseness")
			print(sparseness(x))
			print(np.linalg.norm(x, ord = 1))
			print(L2)
			print(x)

			print("desired sparseness")
			print(spars)
			print(L1)
			print(L2)
			
		
			if sparseness(x) < spars:
				H[k,:] = nn_sparse_constrained_proj(x, L1, L2)
			else:
				H[k,:][x < 0] = 0 

			
			print("new sparseness")
			print(sparseness(x))
			print(np.linalg.norm(x, ord = 1))
			print(np.linalg.norm(x))
			print(x)
			

		print("error :")
		error_ = np.linalg.norm(X - W.dot(H))
		print(error_)
		if np.abs(error_ - error) < 0.0001:
			break
		error = error_
	return W, H

def nmf_lee_and_seung(X, dim):

	step = 5
	m, n = np.shape(X)

	W = np.random.rand(m, dim)
	H = np.random.rand(dim, n)
	error = np.inf

	while True:
		W = np.nan_to_num(lee_W(X, W, H))
		H = np.nan_to_num(lee_H(X, W, H))

		print("error :")
		error_ = np.linalg.norm(X - W.dot(H))
		print(error_)
		if np.abs(error_ - error) < 0.0001:
			break
		error = error_
	return W, H



def merge_index(V,I,C, thresh):
		#thresh = cluster_thresh
		MergeFits = defaultdict(list)
		for a in V:
			if not np.equal(a.all(), 0):
				if I.shape[0] > 0:
					fits = distance.cdist([a],I,'cosine')[0]
					clust = fits.argsort()[0]
					if fits[clust] < 1-thresh:
						MergeFits[clust].append(a)
					else:
						I = np.concatenate((I,[a]))
						C[len(C)] = 1
				else:
					I = np.array([a])
					C[0] = 1
		for k,v in MergeFits.items():
			I[k,:] = np.concatenate(([I[k,:]*C[k]],v)).sum(0)/(len(v)+C[k])
			C[k] += len(v)
		return C,I

def collapse_index(I,C, combine_thresh):
	remove_clusters = {}
	D = distance.pdist(I,'cosine')
	D = D < (1 - combine_thresh)
	i = 0
	j = 1
	for d in D:
		if j >= I.shape[0]:
			i += 1
			j = i+1
		if d:
			if C[i] >= C[j]:
				remove_clusters[j] = True
			else:
				remove_clusters[i] = True
		j += 1
	Cnew = {}
	for i in range(len(C)):
		if i not in remove_clusters:
			Cnew[len(Cnew)] = C[i]
	return Cnew,I[[i for i in range(len(C)) if i not in remove_clusters],:]


def lsi_cluster_index(vectors, thresh,  random_chunk = 0.002, cluster_iters = 200):
	Clusters = {}
	hash_size = 22
	dim = np.shape(vectors)[1]
	Index = np.zeros((0, dim))
	chunk_size = random_chunk*2**hash_size
	print(chunk_size)
	print(dim)
	for ci in range(cluster_iters):
		l = np.random.randint( 0, 2**hash_size - chunk_size)
		seed_vectors = vectors[l:(chunk_size + l),:]
		Clusters, Index = merge_index(seed_vectors, Index, Clusters, thresh)
		Clusters, Index = collapse_index(Index, Clusters, thresh)
		#print(np.shape(Index))
		print ci,len(Clusters)
	return Index

def lsi_cluster_part(vectors, Index, cluster_thresh):

	Clusters = [np.empty((np.shape(vectors)[0],),dtype=np.int64) for _ in range(Index.shape[0])]
	Sizes = np.zeros(Index.shape[0],dtype=np.int64)
	num_best = 5

	vector_block = []
	block_index = []
	count = 0
	count_null  = 0
	for col,a in enumerate(vectors):
		#print(col)
		# this is slow and not particularly clever...
		#if not np.not_equal(a.any(), 0):
		if a.any() != 0.0:
			count += 1
			
			block_index.append(col)
			vector_block.append(a)
			
		else:

			count_null += 1


		if len(vector_block) == 10**4:
			D = distance.cdist(vector_block, Index, 'cosine')
			for indexed_doc in enumerate(D):
				colx,fits = indexed_doc
				MI = fits.argsort()[:num_best]
				for clust in MI:
					#print(fits[clust])
					if fits[clust] < 1 - cluster_thresh:
						Clusters[clust][Sizes[clust]] = block_index[colx]
						Sizes[clust] += 1
					else:
						break
			block_index = []
			vector_block = []
	if len(vector_block) > 0:
		D = distance.cdist(vector_block, Index, 'cosine')
		for indexed_doc in enumerate(D):
			colx,fits = indexed_doc
			MI = fits.argsort()[:num_best]
			for clust in MI:
				#print(fits[clust])
				if fits[clust] < 1 - cluster_thresh:
					Clusters[clust][Sizes[clust]] = block_index[colx]
					Sizes[clust] += 1
				else:
					break

	print(count)
	print(count_null)
	return [c[:Sizes[i]] for i,c in enumerate(Clusters)]




mat_name = "abundance_cohort.npy"

try:
	args = sys.argv[1:]
	mat_name = args[0]
	thres1 = float(args[1])
	thres2 = float(args[2])
	print(mat_name, thres1, thres2)
except:
	thres1 = 0.8
	thres2 = 0.8
	pass


abundance = np.load(mat_name)

print(np.shape(abundance))

I, U = svd(abundance)
I = np.diag(I)

V = np.linalg.pinv(I).dot(U.transpose()).dot(abundance)

np.save("svd_cohort", V)

"""
V = np.linalg.pinv(I).dot(np.linalg.pinv(U)).dot(abundance)
"""


X = U.dot(I).dot(V)

print(np.shape(X))
print(np.max(X - abundance))

print(np.shape(V))

#vectors = V.transpose().dot(I)

vectors = abundance.transpose()


"""
W, H = nmf_lee_and_seung(abundance, 18)

np.save("W_lee", W)
np.save("H_lee", H)
"""



Index = lsi_cluster_index(vectors, thres1)
print(np.shape(Index))
clusters = lsi_cluster_part(vectors, Index, thres2)



print("save data")
np.save("index_cohort", Index)
np.save("clusters_cohort", clusters)


#np.savetxt("index.txt", Index, delimiter=' ')
#np.savetxt("clusters.txt", clusters, delimiter=' ')


#print(lsi_cluster_part(vectors, Index, thresh))
#print(svd(abundance))

#print(nmf(abundance))

#u, I = rank1(C)
#print(u, I)


"""

Ab = np.array([[1, 1, 0, 0, 2, 4], [0, 0, 2, 1, 1, 2]])
Ab2 = np.array([[1, 1, 0, 0, 2, 4], [0, 0, 2, 1, 1, 2], [2, 2, 6, 3, 4, 8]])


print(svd(Ab2))
print(nmf(Ab2))


C = Ab2.dot(np.transpose(Ab2))
print(C)

S = 3

U = np.zeros((S,S))
I = np.zeros(S)
Id = np.eye(3)

for k in range(0, S):

	u, i = rank1(C)
	U[:,k] = u
	T = Id - U.dot(np.linalg.pinv(U))
	R = T.dot(Ab2)
	R[R < 0] = 0
	C = R.dot(R.transpose())

	
	Z = np.zeros((3,3))
	Z[:,k] = u
	I[k] = i


print(C)
print(U)
print(np.sqrt(I))
print(np.linalg.inv(U).dot(Ab2))
print(np.transpose(U).dot(Ab2))


svd = np.linalg.svd(Ab2, full_matrices=0)

print(svd[0])
print(svd[1])
print(svd[2])

"""



