This code, dsgd_mf.py, runs using Spark, Python 2.7, and the numpy package for Python.  Its purpose is
to read in .csv data representing a sparse matrix of values (described more below) and factorize that
data into two matrices.  It was written for HW 7 for the class 10-605 at Carnegie Mellon, Spring 2015,
by me, Benjamin Steele, in April 2015.

To use this code, type the following command into the command line when in the code's folder:

spark-submit dsgd_mf.py num_factors num_workers num_iterations beta_value lambda_value inputV_filepath outputW_filepath outputH_filepath

Here, num_factors represents the number of different factors you wish to use to represent the dataset
after factorization, num_workers represents the number of different parallel jobs you wish to run to
calculate the factorization, and beta and lambda are specific values governing the convergence rate and
regularization of the stochastic gradient descent algorithm, respectively.

The input filepath to the data is self-explanatory.  Data should be in a csv file, with the first and 
second columns unique IDs and the third columns values for that specific ID pair.

The factorized output matrices will be written to the output W and H filepaths, the last two arguments.