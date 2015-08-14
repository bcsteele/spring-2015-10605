## Benjamin Steele - 4/12/2015
## 10-605 Homework 7
## Distributed SGD for Matrix Factorization on Spark

## Example argument:
## spark-submit dsgd_mf.py 100 10 50 0.8 1.0 autolab_train.csv w.csv h.csv


## Initializing Spark and import statements

import sys
import numpy
import random
from pyspark import SparkContext, SparkConf

conf = SparkConf()#.setAppName("local").setMaster(master)
sc = SparkContext(conf=conf)



######## Function definitions

## Maps input data to user id key, (movie id, rating) RDD.

## Maps input data to (id key, recoded index) value hashmap/dictionary.
def id_mapper(rdd):
	partition_sizes = rdd.mapPartitions(lambda it: [sum(1 for i in it)]).collect()
	partition_starts = [sum(partition_sizes[:i]) for i in range(len(partition_sizes))]
	return rdd.mapPartitionsWithIndex(lambda i,k: ((x, partition_starts[i]+j) for j,x in enumerate(k))).collectAsMap()

## Given transformed user_id, movie_id values, and k (worker number), returns (stratum,(block_key,(transformed_user_id,transformed_movie_id,rating))) values.
## Wraps when appropriate.

def strata_maker(values):
	block_row = keyMixerUser[values[0]]%num_workers
	block_column = keyMixerMovie[values[1]]%num_workers
	stratum = block_column-block_row
	## This ensures "wrap" of the diagonal around the matrix.  Matrix is always square due to our block partitioning into k rows, k columns.
	if stratum < 0:
		stratum += num_workers
	return (stratum, (block_row, values) )

########

## Doing SGD updates with L2 loss.

## Values are in format (partition_id, (block_row, block col, user_id, movie_id, rating))
def blockSGD(values):

	it_num = iteration_number
	## just local aliases, but matrices not passed back to main process so doesn't matter
	local_w_matrix = w_matrix
	local_h_matrix = h_matrix

	users_seen = set()
	movies_seen = set()

	input_block_data = [z[1] for z in values]

	for listing in input_block_data:

		epsilon = (tau_zero+it_num)**(-beta_value)

		user_id = listing[0]
		movie_id = listing[1]
		rating = listing[2]

		users_seen.add(user_id)
		movies_seen.add(movie_id)

		## Has to be copies since otherwise will change value by h_col update.
		w_row = numpy.copy(local_w_matrix[user_id,:])
		h_col = numpy.copy(local_h_matrix[:,movie_id])

		grad = -2*(rating - numpy.dot(w_row,h_col))

		local_w_matrix[user_id,:] -= epsilon*(grad*h_col + 2*lambda_value*w_row/records_per_user[user_id])
		local_h_matrix[:,movie_id] -= epsilon*(grad*w_row + 2*lambda_value*h_col/records_per_movie[movie_id])

		it_num += 1

	w_changes = []
	for user_seen in users_seen:
		w_changes.append(['w',user_seen, local_w_matrix[user_seen,:]])

	h_changes = []
	for movie_seen in movies_seen:
		h_changes.append(['h',movie_seen, local_h_matrix[:,movie_seen]])

	return (it_num-iteration_number,w_changes, h_changes)

def update_matrix(values):
	## Writing updates to w and h matrices.
	if type(values) is int: ## Means it is the iteration count from that worker.
		global iteration_number
		iteration_number += values
	else:
		for part in values:
		
			id = part[1]
			if part[0] == 'w':
				w_matrix[id,:] = part[2]
			else:
				h_matrix[:,id] = part[2]

########



## Reading in command-line arguments

num_factors = int(sys.argv[1])
num_workers = int(sys.argv[2])
num_iterations = int(sys.argv[3])
beta_value = float(sys.argv[4])
lambda_value = float(sys.argv[5])
inputV_filepath = sys.argv[6]
outputW_filepath = sys.argv[7]
outputH_filepath = sys.argv[8]

tau_zero = 100
iteration_number = 0


########

## Reading in the external data and creating a list (RDD).
input_data = sc.textFile(inputV_filepath).map(lambda line: [int(z) for z in line.split(",")])

## Reading the user and movie IDs from this list into a new list (RDD).
id_data = input_data.map(lambda values: (values[0],values[1]))

## Reading user_ids and remapping to (original user_id, recoded id).
user_ids = id_data.keys().distinct()
user_id_map = id_mapper(user_ids)

## Reading movie_ids and remapping to (original movie_id, recoded id).
movie_ids = id_data.values().distinct()
movie_id_map = id_mapper(movie_ids)

## Counting number of users, movies represented in input data.
number_users = user_ids.count()
number_movies = movie_ids.count()

## Calculating N_i* and N_*j, the number of entries in the ith row and nth column, respectively.
records_per_user_old_coding = id_data.countByKey()
records_per_user = dict()
for key, value in records_per_user_old_coding.iteritems():
	records_per_user[user_id_map[key]] = value

records_per_movie_old_coding = id_data.map(lambda values: (values[1],1)).countByKey()
records_per_movie = dict()
for key, value in records_per_movie_old_coding.iteritems():
	records_per_movie[movie_id_map[key]] = value

########

## Rewriting the data as a list of transformed user and movie IDs.  (Mapping back to original IDs is possible with the dictionaries.)
data_indices = input_data.map(lambda values: (user_id_map[values[0]],movie_id_map[values[1]],values[2]))

######## Making block and strata assignments.

keyMixerUser = range(number_users) ## Assigns a random mapping for each ID value.
keyMixerMovie = range(number_movies)
random.shuffle(keyMixerUser)
random.shuffle(keyMixerMovie)

## Returns (stratum,(block_key,(transformed_user_id,transformed_movie_id,rating))) values.
data_strata = data_indices.map(strata_maker)


## Preparing to run SGD: Initializing the matrices W and H with random 0-1 values.
w_matrix = numpy.random.rand(number_users, num_factors)
h_matrix = numpy.random.rand(num_factors, number_movies)


######## This portion is the iterating loop that runs SGD.


for iterations in xrange(num_iterations):

	stratum_to_analyze = 0

	## The active stratum is selected, and is partitioned out into workers along the block row numbers (aka the second key).
	active_stratum_partitioned = data_strata.filter(lambda values: values[0] == stratum_to_analyze).map(lambda values: values[1]).partitionBy(num_workers)

	#prior = w_matrix.sum().sum()

	sgd_results = active_stratum_partitioned.mapPartitions(blockSGD)

	for partition_output in sgd_results.collect():

		#print partition_output

		update_matrix(partition_output)

	#print prior
	#print w_matrix.sum().sum()

	## Need to reassign the strata at random.
	random.shuffle(keyMixerUser)
	random.shuffle(keyMixerMovie)

	data_strata = data_indices.map(strata_maker)



######## Writing the output results to the selected files.
## First step is to map back line identifiers into a non-sparse, and original-ordered, form.
## Second step is to write to .csv file format.
## I assume the IDs are indexed starting at 1.

w_matrix_new = numpy.zeros((user_ids.max(), num_factors))
h_matrix_new = numpy.zeros((num_factors, movie_ids.max()))

for key, value in user_id_map.iteritems():
	w_matrix_new[key-1,:] = w_matrix[value,:]

for key, value in movie_id_map.iteritems():
	h_matrix_new[:,key-1] = h_matrix[:,value]

numpy.savetxt(outputW_filepath, w_matrix_new, delimiter=",")
numpy.savetxt(outputH_filepath, h_matrix_new, delimiter=",")

