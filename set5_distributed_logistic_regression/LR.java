//10-605: Problem Set 5
//Benjamin Steele, 3/16/2015
//
// A program to perform logistic regression using stochastic gradient descent.
// Testing usage is in the form: 
// cat abstract.tiny.train | java -cp . -Xmx128m LR 10000 0.5 0.1 20 1000 abstract.tiny.test
// cat abstract.tiny.train | java -cp . -Xmx128m LR 10000 0.5 0.1 20 1000 abstract.tiny.test


import java.io.*;
import java.util.*;

public class LR {
    public static void main(String [] args) throws IOException {

	// Section 1: Initializing variables for the program.

	Long k = 0L; // counter for example number
	Integer numLabelClasses = 14; // 14 languages

	Integer vocabularySize = Integer.valueOf(args[0]);
	Double learningRate = Double.valueOf(args[1]);
	Double regularizationTerm = Double.valueOf(args[2]);
	Integer maxIteration = Integer.valueOf(args[3]);
	Integer trainingSize = Integer.valueOf(args[4]);
	File testingFile = new File(args[5]); // the test file to load

	// A large double array with 14 rows (1 row/language)
	// and as many columns as the vocabulary hashing size.

	double[][] Blist = new double[numLabelClasses][vocabularySize];
	long[][] Alist = new long[numLabelClasses][vocabularySize];

	// A lookup hashmap relating language label to position.

	Map<String, Integer> languageLookup = new HashMap<String, Integer>();
	languageLookup.put("ca",0);
	languageLookup.put("de",1);
	languageLookup.put("el",2);
	languageLookup.put("es",3);
	languageLookup.put("fr",4);
	languageLookup.put("ga",5);
	languageLookup.put("hu",6);
	languageLookup.put("hr",7);
	languageLookup.put("nl",8);
	languageLookup.put("pl",9);
	languageLookup.put("pt",10);
	languageLookup.put("ru",11);
	languageLookup.put("sl",12);
	languageLookup.put("tr",13);


	// Section 2: Read in many iterations of the training data.
	// Iterate to train the logistic regression classifier.

	// As this program is set up it does not explicitly iterate over the dataset T times.
	// Instead, the output is provided already shuffled and concatenated.

	// Reading the input data line-by-line
	BufferedReader readCounts = null;

	try {

		readCounts = new BufferedReader(new InputStreamReader(System.in)); // reads file from stdin
		String thisLine = null;
		Integer linesRead = 1;

		while ((thisLine = readCounts.readLine()) != null) {

			Integer t = 1 + linesRead/trainingSize; // the number of passes through the training set started
			linesRead += 1;

			learningRate /= t*t;

			String[] parts = thisLine.split("\t",2); // splits on first tab
			String[] labels = parts[0].split(","); // splits language labels apart

			// Vector<String> documentWords = tokenizeDoc(parts[1]);
			
			// Produces a list of the tokenized, then hashed values in the document.
			// A hashbucket is counted once for each document word.
			List<Integer> hashedWords = stringHashing(tokenizeDoc(parts[1]), vocabularySize);


			// Making an array of length 14 with "0" where this label is not present for
			// this document and "1" where this label is present for this document.
			int[] hasLabels = new int[numLabelClasses]; // initializes to 0
			for (String label: labels) {
				hasLabels[languageLookup.get(label)] = 1; // sets appropriate labels to 1
			}



			// Looping over the label classes for each document.
			// Performs the regularization updates.

			Integer counter = 0; // counting which label the update is on
			for (Integer label: hasLabels) {

				k += 1; // incrementing k, the documents*labels*iterations counter

				Integer labelIndex = languageLookup.get(label);

				for (Integer hashWord: hashedWords) {

					// Simulate the regularization updates for the B[j]'s.
					Blist[counter][hashWord] *= Math.pow(1-2*learningRate*regularizationTerm, k-Alist[counter][hashWord]);
					
					// Set the appropriate A[j]'s.
					Alist[counter][hashWord] = k;

				}


				// Calculating p for this document given this label, under existing model B[j]'s

				Double BtransposeX = 0.0;
				for (Integer hashWord: hashedWords) {
					BtransposeX += Blist[counter][hashWord]; // just the value of Bj, since each hashed word has value 1
				}
				//Double p = 1/(Math.exp(-BtransposeX)+1);
				double p = sigmoid(BtransposeX); // using the provided command to prevent overflows


				// Setting B[j] = B[j] + lambda(y-p)x_i
				for (Integer hashWord: hashedWords) {

					// resetting B[j] values
					Blist[counter][hashWord] += learningRate*(label-p); // x_i = 1, since treating words separately

				}

				counter += 1;
				

			}

		}

		// Step 3 in the handout: the final regularization update sweep.
		for (int c=0; c<numLabelClasses; c++) { // for number of rows in Bj
			for (int d=0; d<vocabularySize; d++) { // for number of columns in Bj
				Blist[c][d] *= Math.pow(1-2*learningRate*regularizationTerm, k-Alist[c][d]);
			}
		}


	} catch (FileNotFoundException e) {
	    e.printStackTrace();
	} catch (IOException e) {
	    e.printStackTrace();
	} finally {
	    try {
	        if (readCounts != null) {
	            readCounts.close();
	        }
	    } catch (IOException e) {
	    }
	}




	// Section 3: Reading in the test dataset and outputting predicted labels.
	// Reads in the test set line-by-line (document-by-document), reduces each to
	// hashed features, and outputs all labels and scores.

	BufferedReader reader = null;

	try {

		reader = new BufferedReader(new FileReader(testingFile));
		String thisLine = null;

		while ((thisLine = reader.readLine()) != null) {

			String[] parts = thisLine.split("\t",2); // splits on first tab
			// String[] labels = parts[0].split(","); // splits language labels apart
			
			List<Integer> hashedWords = stringHashing(tokenizeDoc(parts[1]), vocabularySize);


			// Now must iterate to calculate P(class | document) for each class.
			// Iterate over each class and over each feature in the document.

			double[] pList = new double[numLabelClasses];

			for (int c=0; c<numLabelClasses; c++) { // for each of the 14 language classes

				Double BtransposeX = 0.0;

				for (Integer hashWord: hashedWords) { // for each feature in this test document
					BtransposeX += Blist[c][hashWord];
				}
				pList[c] = sigmoid(BtransposeX); // setting this class's computed p-value

			}

			System.out.println("nl\t"+pList[8]+",el\t"+pList[2]+",ru\t"+pList[11]+",sl\t"+pList[12]+",pl\t"+pList[9]+",ca\t"+pList[0]+",fr\t"+pList[4]+
				",tr\t"+pList[13]+",hu\t"+pList[6]+",de\t"+pList[1]+",hr\t"+pList[7]+",es\t"+pList[3]+",ga\t"+pList[5]+",pt\t"+pList[10]);

		}


	} catch (FileNotFoundException e) {
	    e.printStackTrace();
	} catch (IOException e) {
	    e.printStackTrace();
	} finally {
	    try {
	        if (reader != null) {
	            reader.close();
	        }
	    } catch (IOException e) {
	    }
	}
    }


	// Method for changing a document into word features
	public static Vector<String> tokenizeDoc(String cur_doc) {	        String[] words = cur_doc.split("\\s+");	        Vector<String> tokens = new Vector<String>();	        for (int i = 0; i < words.length; i++) {
			words[i] = words[i].replaceAll("\\W", "");			if (words[i].length() > 0) {
				tokens.add(words[i]);			}
		}		return tokens;
	}

	// Method for hashing Vector<String> words into a set number of hash buckets.
	public static List<Integer> stringHashing(Vector<String> documentWords, Integer numHashBuckets) {
		List<Integer> hashed = new ArrayList<Integer>();
		for (String thisWord: documentWords) {
			// Hashing the word into the range [0, vocabSize-1].
			hashed.add(Math.abs(thisWord.hashCode() % numHashBuckets)); 
		}

		return hashed;
	}

	// Method for calculating sigmoid probability, avoiding overflow.  Provided in HW description.
	public static double overflow=20;	public static double sigmoid(double score) {		if (score > overflow) score = overflow;		else if (score < -overflow) score = -overflow;		// double exp = Math.exp(score);		return 1 / (1 + Math.exp(-score));	}


}
