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

def svd(X, dim = "full_rank"):

	C = X.dot(np.transpose(X))

	S = np.shape(C)[0]

	if dim == "full_rank":
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



def merge_index(V,I,C, thresh):
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

	return [c[:Sizes[i]] for i,c in enumerate(Clusters)]


def max_log_lik_ratio(s, bkg, h1_prob=0.8,thresh1=3.84,thresh2=np.inf):
	LLR = [(None,None)]
	read_match_sum = s[-1]
	del s[-1]
	v1 = read_match_sum*h1_prob*(1-h1_prob)
	m1 = read_match_sum*h1_prob
	for k,sect_sum in s.items():
		if sect_sum > read_match_sum*bkg[k]:
			v2 = read_match_sum*bkg[k]*(1-bkg[k])
			m2 = read_match_sum*bkg[k]
			llr = np.log(v2**.5/v1**.5) + .5*((sect_sum-m2)**2/v2 - (sect_sum-m1)**2/v1)
			LLR.append((llr,k))
	LLR.sort(reverse=True)
	K = []
	if LLR[0][0] > thresh1:
		K.append(LLR[0][1])
	for llr,k in LLR[1:]:
		if llr > thresh2:
			K.append(k)
		else:
			break
	return K

def hash_read_generator(file_object,max_reads=10**15,newline='\n'):
	line = file_object.readline().strip()
	lastlinechar = ''
	read_strings = []
	r = 0
	while (line != '') and (r < max_reads):
		# read_id (always) and quality (sometimes) begin with '@', but quality preceded by '+' 
		if (line[0] == '@') and (lastlinechar != '+'):
			if len(read_strings) == 5:
				try:
					I = newline.join(read_strings[:-1])
					B = np.fromstring(read_strings[-1][10:-2],dtype=np.uint64,sep=',')
					yield (I,B[0],B[1:])
				except Exception,err:
					print str(err)
				r += 1
			read_strings = []
		read_strings.append(line)
		lastlinechar = line[0]
		line = file_object.readline().strip()
			



print("load abundance matrix")

input_path = "hashed_reads/"
Kmer_Hash_Count_Files = glob.glob(os.path.join(input_path,'*.count.hash.conditioned'))

abundance = []
for f in Kmer_Hash_Count_Files:
	m = np.fromfile(f,dtype=np.float32)
	abundance.append(m)

abundance = np.array(abundance)
np.save("abundance", abundance)


print("compute SVD")

I, U = svd(abundance)
I = np.diag(I)
V = np.linalg.pinv(I).dot(U.transpose()).dot(abundance)
vectors = V.transpose()


print("cluster eigen-genomes")

thres1 = 0.6
thres2 = 0.6

Index = lsi_cluster_index(vectors, thres1)
clusters = lsi_cluster_part(vectors, Index, thres2)


print("save data")
np.save("index", Index)
np.save("clusters", clusters)


####    cluster_cols


hash_size = 22
I = Index.shape[0]
cluster_sizes = np.zeros(I, dtype=np.uint64)


GW = np.load('global_weights.npy')
global_weight_sum = GW.sum()

CP = np.zeros(I)
X = np.zeros((2**hash_size,5),dtype=np.int16)
Ix = np.zeros(2**hash_size,dtype=np.int8)

for i,c in enumerate(clusters):
	CP[i] = GW[c].sum()/global_weight_sum
	cluster_sizes[i] = c.shape[0]
	X[c,Ix[c]] = i + 1
	Ix[c] += 1




np.save("cluster_cols", X)
np.save('cluster_probs', CP)
np.save('kmer_cluster_sizes', cluster_sizes)


####  write_partition_part
print("partition reads")

cluster_probs = dict(enumerate(CP))


Hashq_Files = glob.glob(os.path.join('/export/home/vprost/workspace/NMF/hashed_reads/','*.hashq.*'))
Hashq_Files.sort()

reads_written = 0
unique_reads_written = 0
count = 0
output_path = "clusters/"


for fr in range(0, len(Hashq_Files)):
	infile = Hashq_Files[fr]
	f = gzip.open(infile)
	sample_id = infile[infile.rfind('/')+1:infile.index('.hashq')]
	sample_id_nb = sample_id[3:9]

	clusters = np.load("cluster_cols.npy")
	values = np.load("global_weights.npy")

	CF = {}
	

	for a in hash_read_generator(f):
		
		count += 1
		spike = 0

		sys.stdout.write("read partitionned : %d   \r" % count )
		sys.stdout.flush()

		name = a[0]
		name = name[0:name.find(' ')]

		if name[0:6] == '@Spike':
			spike = 1
		

		name = name[name.find('.') + 1:]

		kmers = a[2]
		kmers = [int(kmer) for kmer in kmers]
		D = defaultdict(float)
		for kmer in kmers:

			c = clusters[kmer,:]
			c = c[np.nonzero(c)[0]] - 1
			if len(c) > 0:
				D[-1] += values[kmer]
				for clust in c:
					D[clust] +=  values[kmer]
		best_clusts = max_log_lik_ratio(D, cluster_probs)


		for best_clust in best_clusts:
			if best_clust not in CF:
				try:
					CF[best_clust] = open('%s%d_reads_id' % (output_path, best_clust),'a')
				except:
					CF[best_clust] = open('%s%d_reads_id' % (output_path, best_clust),'a')
			CF[best_clust].write(sample_id_nb + ' ' + name + ' ' + str(spike) +'\n')
			reads_written += 1
		if len(best_clusts) > 0:
			unique_reads_written += 1
		if len(CF) > 200:
			for cfv in CF.values():
				cfv.close()
			CF = {}

for f in CF.values():
	f.close()

print 'total reads written:',reads_written
print 'unique reads written:',unique_reads_written