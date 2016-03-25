/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.transform;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.JobReturn;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;


@SuppressWarnings("deprecation")
public class ApplyTfCSVMR {
	
	public static JobReturn runJob(String inputPath, String spec, String mapsPath, String tmpPath, String outputPath, String partOffsetsFile, CSVFileFormatProperties inputDataProperties, long numCols, int replication, String headerLine) throws IOException, ClassNotFoundException, InterruptedException {
		JobConf job = new JobConf(ApplyTfCSVMR.class);
		job.setJobName("ApplyTfCSV");

		/* Setup MapReduce Job */
		job.setJarByClass(ApplyTfCSVMR.class);
		
		// set relevant classes
		job.setMapperClass(ApplyTfCSVMapper.class);
		job.setNumReduceTasks(0);
	
		// Add transformation metadata file as well as partOffsetsFile to Distributed cache
		DistributedCache.addCacheFile((new Path(mapsPath)).toUri(), job);
		DistributedCache.createSymlink(job);
		
		Path cachefile=new Path(partOffsetsFile);
		DistributedCache.addCacheFile(cachefile.toUri(), job);
		DistributedCache.createSymlink(job);
		
		// set input and output properties
		job.setInputFormat(TextInputFormat.class);
		job.setOutputFormat(TextOutputFormat.class);
		
		job.setMapOutputKeyClass(NullWritable.class);
		job.setMapOutputValueClass(Text.class);
		
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);
		
		job.setInt(MRConfigurationNames.DFS_REPLICATION, replication);
		
		FileInputFormat.addInputPath(job, new Path(inputPath));
		// delete outputPath, if exists already.
		Path outPath = new Path(outputPath);
		FileSystem fs = FileSystem.get(job);
		fs.delete(outPath, true);
		FileOutputFormat.setOutputPath(job, outPath);

		job.set(MRJobConfiguration.TF_HAS_HEADER, 	Boolean.toString(inputDataProperties.hasHeader()));
		job.set(MRJobConfiguration.TF_DELIM, 		inputDataProperties.getDelim());
		if ( inputDataProperties.getNAStrings() != null)
			// Adding "dummy" string to handle the case of na_strings = ""
			job.set(MRJobConfiguration.TF_NA_STRINGS, TfUtils.prepNAStrings(inputDataProperties.getNAStrings()) );
		job.set(MRJobConfiguration.TF_SPEC, spec);
		job.set(MRJobConfiguration.TF_SMALLEST_FILE, CSVReblockMR.findSmallestFile(job, inputPath));
		job.set(MRJobConfiguration.OUTPUT_MATRICES_DIRS_CONFIG, outputPath);
		job.setLong(MRJobConfiguration.TF_NUM_COLS, numCols);
		job.set(MRJobConfiguration.TF_TXMTD_PATH, mapsPath);
		job.set(MRJobConfiguration.TF_HEADER, headerLine);
		job.set(CSVReblockMR.ROWID_FILE_NAME, cachefile.toString());
		job.set(MRJobConfiguration.TF_TMP_LOC, tmpPath);
		
		//turn off adaptivemr
		job.setBoolean("adaptivemr.map.enable", false);

		// Run the job
		RunningJob runjob = JobClient.runJob(job);
		
		// Since transform CSV produces part files w/ prefix transform-part-*,
		// delete all the "default" part-..... files
		deletePartFiles(fs, outPath);
		
		MatrixCharacteristics mc = new MatrixCharacteristics();
		return new JobReturn(new MatrixCharacteristics[]{mc}, runjob.isSuccessful());
	}
	
	private static void deletePartFiles(FileSystem fs, Path path) throws FileNotFoundException, IOException
	{
		PathFilter filter=new PathFilter(){
			public boolean accept(Path file) {
				return file.getName().startsWith("part-");
	        }
		};
		FileStatus[] list = fs.listStatus(path, filter);
		for(FileStatus stat : list) {
			fs.delete(stat.getPath(), false);
		}
	}
	
}
