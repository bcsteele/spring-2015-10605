// 10-605 Assignment 3: Naive Bayes with Hadoop MapReduce
// Benjamin Steele - 2/25/2015

// This program, run.java, takes as input three arguments:
// (1) an InputPath where data is present
// (2) an OutputPath where finished results will be written to
// (3) the number of reduce tasks to be performed.  For this program
//     mapping and reduction are performed by the same program.

//package NB_on_hadoop;

import java.io.*;
import java.util.*;
import java.util.regex.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class NB_train_hadoop {

	// Code copied from Hadoop-MapReduce class lecture slides, 1/27/15
	// "MR code: Word Count Map"

	public static class Map extends Mapper<LongWritable, Text, Text, LongWritable>{
		// declaring the counters used to record occurrences
		Long numberTrainingInstances = 0L;
		java.util.Map<String, Long> labelCountMap = new HashMap<String, Long>();
		java.util.Map<String, Long> featureSumByLabelMap = new HashMap<String, Long>();
		java.util.Map<String, Long> featureCountMap = new HashMap<String, Long>();

		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

			numberTrainingInstances++; // incrementing the number of instances
			String[] parts = value.toString().split("\t",2); // splits on first tab
			Vector<String> tokenized = tokenizeDoc(parts[1]); // tokenizing the body text of the document

			if (featureCountMap.size() > 500) { // if over 500 entries

				for (java.util.Map.Entry<String,Long> record : featureCountMap.entrySet()) {
					Text wordInformation = new Text("Y="+record.getKey());
					LongWritable countValue = new LongWritable(record.getValue());
					context.write(wordInformation,countValue);
				}
				featureCountMap.clear();
			}


			// finding the language classes that apply to this document
			Matcher m = Pattern.compile("\\w*CAT\\b").matcher(parts[0]);
			while(m.find()) {
			// for (String thisClass : parts[0].split(",")) {  // the labels for that document
				String thisClass = m.group();
			
				// updating the appropriate class count, total word count for that class
				tableUpdateCount(featureSumByLabelMap, thisClass, tokenized.size());
				tableUpdateCount(labelCountMap,thisClass,1);

				// adding the tokenizer word counts for this class
				for(String thisWord: tokenized){
					tableUpdateCount(featureCountMap,thisClass+",W="+thisWord,1);
				}
			}
		}


		public void cleanup(Context context) throws IOException, InterruptedException {
			// sends remaining hashmaps out to context as the final output step

			for (java.util.Map.Entry<String,Long> record : featureCountMap.entrySet()) {
				Text wordInformation = new Text("Y="+record.getKey());
				LongWritable countValue = new LongWritable(record.getValue());
				context.write(wordInformation,countValue);
			}

			// printing out the total number of features (words) for each class
			for (java.util.Map.Entry<String,Long> entry : featureSumByLabelMap.entrySet()) {
				Text wordClassCountDesc = new Text("Y="+entry.getKey()+",W=*");
				LongWritable wordClassCount = new LongWritable(entry.getValue());
				context.write(wordClassCountDesc,wordClassCount);
			}

			// printing out the number of documents in each class
			for (java.util.Map.Entry<String,Long> entry : labelCountMap.entrySet()) {
				Text whichClass = new Text("Y="+entry.getKey());
				LongWritable numberInClass = new LongWritable(entry.getValue());
				context.write(whichClass,numberInClass);
			}

			// printing the overall number of training documents
			Text docNumTotal = new Text("Y=*");
			LongWritable trainingInstances = new LongWritable(numberTrainingInstances);
			context.write(docNumTotal,trainingInstances);

		}
	}

	// Code copied from Hadoop-MapReduce class lecture slides, 1/27/15
	// "MR code: Word count Reduce"

	public static class Reduce extends Reducer<Text, LongWritable, Text, LongWritable>{
		public void reduce(Text key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			long sum = 0L;
			for (LongWritable val : values) {
				sum += val.get();
			}
			context.write(key, new LongWritable(sum));

		}
	}


	// tokenizer: method for changing a document into word features
	// provided in Homework 1 and reused here
	public static Vector<String> tokenizeDoc(String cur_doc) {
		String[] words = cur_doc.split("\\s+");	        Vector<String> tokens = new Vector<String>();	        for (int i = 0; i < words.length; i++) {
			words[i] = words[i].replaceAll("\\W", "");
			if (words[i].length() > 0) {
				tokens.add(words[i]);			}
		}		return tokens;
	}



	// method for updating a hash table given key and count
	public static java.util.Map<String, Long> tableUpdateCount(java.util.Map<String, Long> prevMap, String thisKey, Integer thisCount) {
		Long numOccurrence = prevMap.get(thisKey);
		if(numOccurrence == null){
			prevMap.put(thisKey, Long.valueOf(thisCount));
		} else{
			prevMap.put(thisKey, numOccurrence+Long.valueOf(thisCount));
		}
		return prevMap;
	}


}
