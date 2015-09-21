/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.transform;
import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.lib.NullOutputFormat;

import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;

/**
 * MR Job to Generate Transform Metadata based on a given transformation specification file (JSON format).
 *
 */

public class GenTfMtdMR {

	public static final String DELIM = ",";

	public static long runJob(String inputPath, String txMtdPath, String specFileWithIDs, String smallestFile, String partOffsetsFile, CSVFileFormatProperties inputDataProperties, long numCols, int replication, String headerLine) throws IOException, ClassNotFoundException, InterruptedException {
		JobConf job = new JobConf(GenTfMtdMR.class);
		job.setJobName("GenTfMTD");
		
		/* Setup MapReduce Job */
		job.setJarByClass(GenTfMtdMR.class);
		
		// set relevant classes
		job.setMapperClass(GTFMTDMapper.class);
		job.setReducerClass(GTFMTDReducer.class);
	
		// set input and output properties
		job.setInputFormat(TextInputFormat.class);
		job.setOutputFormat(NullOutputFormat.class);
		
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(DistinctValue.class);
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(LongWritable.class);
		
		job.setInt("dfs.replication", replication);

		FileInputFormat.addInputPath(job, new Path(inputPath));
		// delete outputPath, if exists already.
		Path outPath = new Path(txMtdPath);
		FileSystem fs = FileSystem.get(job);
		fs.delete(outPath, true);
		FileOutputFormat.setOutputPath(job, outPath);

		job.set(MRJobConfiguration.TF_HAS_HEADER, Boolean.toString(inputDataProperties.hasHeader()));
		job.set(MRJobConfiguration.TF_DELIM, inputDataProperties.getDelim());
		if ( inputDataProperties.getNAStrings() != null)
			// Adding "dummy" string to handle the case of na_strings = ""
			job.set(MRJobConfiguration.TF_NA_STRINGS, inputDataProperties.getNAStrings() + DataExpression.DELIM_NA_STRING_SEP + "dummy");
		job.set(MRJobConfiguration.TF_SPEC_FILE, specFileWithIDs);
		job.set(MRJobConfiguration.TF_SMALLEST_FILE, smallestFile);
		job.setLong(MRJobConfiguration.TF_NUM_COLS, numCols);
		job.set(MRJobConfiguration.TF_HEADER, headerLine);
		
		job.set(MRJobConfiguration.OUTPUT_MATRICES_DIRS_CONFIG, txMtdPath);
		
		// offsets file to store part-file names and offsets for each input split
		job.set(MRJobConfiguration.TF_OFFSETS_FILE, partOffsetsFile);
		
		//turn off adaptivemr
		job.setBoolean("adaptivemr.map.enable", false);
		
		// Run the job
		RunningJob runjob = JobClient.runJob(job);
		
		Counters c = runjob.getCounters();
		long tx_numRows = c.findCounter(MRJobConfiguration.DataTransformCounters.TRANSFORMED_NUM_ROWS).getCounter();

		return tx_numRows;
	}
	
}
