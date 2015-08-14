This code trains multiple simultaneous regularized logistic regression models using stochastic gradient descent to do so in a memory-efficient fashion.  Input documents are based on Wikipedia article abstracts, and take the form:

pt,tr,hu,es,ru,pl,ca,nl,sl,fr,de,hr,el	chemotherapy sometimes cancer chemotherapy treatment cancer

where the two-letter comma-separated labels preceding the tab represent different non-English languages the corresponding Wikipedia article was translated into.  There are exactly 14 languages for this assignment, and so the program is hardcoded to train 14 logistic regression classifiers, and on test documents will output 14 separate predictions.

To further reduce memory usage, document words are hashed into a dictionary of size N (provided as an argument to the program).  For complete details on the background of the problem, algorithm, and data files for training/testing, see the assignment document:
http://www.andrew.cmu.edu/user/amaurya/docs/10605/homework5.pdf