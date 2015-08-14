This repository contains code written for Carnegie Mellon University's course 10-605, "Machine Learning 
with Large Datasets," taught Spring 2015.  Three different assignments from the course are represented 
in three separate subdirectories within this repository.  Each assignment was based around implementing
and describing a machine learning algorithm in the context of a parallelized, memory-limited 
environment.

The first subdirectory, set3_hadoop_naive_bayes, corresponds to a training a Naive Bayes classifier 
using a Hadoop MapReduce implementation programmed in Java.  The second, 
set5_distributed_logistic_regression, contains a memory-limited implementation of training a logistic 
regression classifier using stochastic gradient descent.  The last, set7_spark_matrix_factorization,
uses Apache Spark to factorize a sparse matrix in a distributed fashion using stochastic gradient 
descent.

Though the great majority of code in these files represents my work, a small amount does not since some
utility functions were provided as part of the assignments.  For more detail on the nature of the 
assignments and the context of the class, see the course webpage at:
http://curtis.ml.cmu.edu/w/courses/index.php/Machine_Learning_with_Large_Datasets_10-605_in_Spring_2015