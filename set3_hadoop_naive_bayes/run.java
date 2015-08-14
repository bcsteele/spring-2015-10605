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

public class run {

	// Code copied from Hadoop-MapReduce class lecture slides, 1/27/15
	// "MR code: Word count Main"

	public static void main(String[] rawArgs) throws Exception {
		GenericOptionsParser parser = new GenericOptionsParser(rawArgs);
		String[] args = parser.getRemainingArgs();

		Configuration conf = new Configuration();
		Job job = new Job(conf, "wordcount");

		/* Tell Hadoop where to locate the code that must be shipped if this
		 * job is to be run across a cluster. Unless the location of code
		 * is specified in some other way (e.g. the -libjars command line
		 * option), all non-Hadoop code required to run this job must be
		 * contained in the JAR containing the specified class (WordCountMap 
		 * in this case).
		 */
		job.setJarByClass(NB_train_hadoop.Map.class);

		job.setMapperClass(NB_train_hadoop.Map.class);
		job.setReducerClass(NB_train_hadoop.Reduce.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(LongWritable.class);

		job.setNumReduceTasks(Integer.parseInt(args[2])); // setting number of reducers

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.waitForCompletion(true);
	}

}
