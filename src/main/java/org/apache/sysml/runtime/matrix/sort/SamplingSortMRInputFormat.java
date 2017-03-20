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

package org.apache.sysml.runtime.matrix.sort;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileRecordReader;
import org.apache.hadoop.util.IndexedSortable;
import org.apache.hadoop.util.QuickSort;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.data.Converter;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;


@SuppressWarnings("rawtypes")
public class SamplingSortMRInputFormat<K extends WritableComparable, V extends Writable> 
extends SequenceFileInputFormat<K,V> 
{
	
	public static final String PARTITION_FILENAME = "_partition.lst";
	static final String SAMPLE_SIZE = "sort.partitions.sample";
	private static JobConf lastConf = null;
	private static InputSplit[] lastResult = null;
	
	public static final String TARGET_KEY_CLASS="target.key.class";
	public static final String TARGET_VALUE_CLASS="target.value.class";
	
	public static void setTargetKeyValueClasses(JobConf job, Class<? extends WritableComparable> keyClass, 
			Class<? extends Writable> valueClass)
	{
		job.setClass(TARGET_KEY_CLASS, keyClass, WritableComparable.class);
		job.setClass(TARGET_VALUE_CLASS, valueClass, Writable.class);
	}
  	
	@Override
	public RecordReader<K,V> getRecordReader(InputSplit split,JobConf job, Reporter reporter)
        throws IOException 
    {
		if(reporter!=null)
			reporter.setStatus(split.toString());
		return new SequenceFileRecordReader<K,V>(job, (FileSplit) split);
    }

	/**
	 * Use the input splits to take samples of the input and generate sample
	 * keys. By default reads 100,000 keys from 10 locations in the input, sorts
	 * them and picks N-1 keys to generate N equally sized partitions.
	 * 
	 * @param conf the job to sample
	 * @param partFile where to write the output file to
	 * @return index value
	 * @throws IOException if something goes wrong
	 * @throws InstantiationException if InstantiationException occurs
	 * @throws IllegalAccessException if IllegalAccessException occurs
	 */
	@SuppressWarnings({ "unchecked", "unused", "deprecation" })
	public static int writePartitionFile(JobConf conf, Path partFile) 
	  throws IOException, InstantiationException, IllegalAccessException
	{    
		SamplingSortMRInputFormat inFormat = new SamplingSortMRInputFormat();
	    Sampler sampler = new Sampler();
	       
	    Class<? extends WritableComparable> targetKeyClass;
		targetKeyClass=(Class<? extends WritableComparable>) conf.getClass(TARGET_KEY_CLASS, WritableComparable.class);
	    //get input converter information
		int brlen = MRJobConfiguration.getNumRowsPerBlock(conf, (byte) 0);
		int bclen = MRJobConfiguration.getNumColumnsPerBlock(conf, (byte) 0);
		
	    //indicate whether the matrix value in this mapper is a matrix cell or a matrix block
	    int partitions = conf.getNumReduceTasks();
	    
	    long sampleSize = conf.getLong(SAMPLE_SIZE, 1000);
	    InputSplit[] splits = inFormat.getSplits(conf, conf.getNumMapTasks());
	    int samples = Math.min(10, splits.length);
	    long recordsPerSample = sampleSize / samples;
	    int sampleStep = splits.length / samples;
	    // take N samples from different parts of the input
	    
	    int totalcount = 0;
	    for(int i=0; i < samples; i++) {
	    	SequenceFileRecordReader reader = 
	    		(SequenceFileRecordReader) inFormat.getRecordReader(splits[sampleStep * i], conf, null);
	    	int count=0;
	    	WritableComparable key = (WritableComparable) reader.createKey();
	    	Writable value = (Writable) reader.createValue();
	    	while (reader.next(key, value) && count<recordsPerSample) {
	    		Converter inputConverter = MRJobConfiguration.getInputConverter(conf, (byte) 0);
	    		inputConverter.setBlockSize(brlen, bclen);
	    		inputConverter.convert(key, value);
	    		while(inputConverter.hasNext())
	    		{
	    			Pair pair=inputConverter.next();
	    			if( pair.getKey() instanceof DoubleWritable ){
	    				sampler.addValue(new DoubleWritable(((DoubleWritable)pair.getKey()).get()));		
	    			}
	    			else if( pair.getValue() instanceof MatrixCell ) {
	    				sampler.addValue(new DoubleWritable(((MatrixCell)pair.getValue()).getValue()));				
	    			}
	    			else	
	    				throw new IOException("SamplingSortMRInputFormat unsupported key/value class: "+pair.getKey().getClass()+":"+pair.getValue().getClass());
	    				
	    			count++;
	    		}
	    		key = (WritableComparable) reader.createKey();
	    		value = (Writable) reader.createValue();
	    	}
	    	totalcount += count;
	    }
	   
	    if( totalcount == 0 ) //empty input files
	    	sampler.addValue(new DoubleWritable(0));
	    
	    FileSystem outFs = partFile.getFileSystem(conf);
	    if (outFs.exists(partFile)) {
	    	outFs.delete(partFile, false);
	    }
	    
	    //note: key value always double/null as expected by partitioner
	    SequenceFile.Writer writer = null;
	    int index0 = -1;
	    try {
		    writer = SequenceFile.createWriter(outFs, conf, partFile, DoubleWritable.class, NullWritable.class);
		    NullWritable nullValue = NullWritable.get();
		    int i = 0;
		    boolean lessthan0=true;
		    for(WritableComparable splitValue : sampler.createPartitions(partitions)) 
		    {
		    	writer.append(splitValue, nullValue);
		    	if(lessthan0 && ((DoubleWritable)splitValue).get()>=0) {
		    		index0=i;
		    		lessthan0=false;
		    	}
		    	i++;
		    }
		    if(lessthan0)
		    	index0=partitions-1;
	    }
	    finally {
	    	IOUtilFunctions.closeSilently(writer);
	    }
	    
	    return index0;
	  }

	@Override
	public InputSplit[] getSplits(JobConf conf, int splits) throws IOException {
		if (conf == lastConf) {
			return lastResult;
		}
		lastConf = conf;
		lastResult = super.getSplits(conf, splits);
		return lastResult;
	}
	

	private static class Sampler implements IndexedSortable 
	{
		private ArrayList<WritableComparable> records = new ArrayList<WritableComparable>();
		
		@SuppressWarnings("unchecked")
		public int compare(int i, int j) {
			WritableComparable left =  records.get(i);
			WritableComparable right = records.get(j);
			return left.compareTo(right);
		}

		public void swap(int i, int j) {
			WritableComparable left = records.get(i);
			WritableComparable right = records.get(j);
			records.set(j, left);
			records.set(i, right);
		}
		
		public void addValue(WritableComparable r)
		{
			records.add(r);
		}
		
		public String toString()
		{
			return records.toString();
		}

	/**
	 * Find the split points for a given sample. The sample keys are sorted
	 * and down sampled to find even split points for the partitions. The
	 * returned keys should be the start of their respective partitions.
	 * @param numPartitions the desired number of partitions
	 * @return an array of size numPartitions - 1 that holds the split points
	 */
		public ArrayList<WritableComparable> createPartitions(int numPartitions) 
		{
			int numRecords = records.size();
			if (numPartitions > numRecords) {
				throw new IllegalArgumentException
				("Requested more partitions than input keys (" + numPartitions +
						" > " + numRecords + ")");
			}
			new QuickSort().sort(this, 0, records.size());
			//System.out.println("after sort: "+ toString());
			float stepSize = numRecords / (float) numPartitions;
			//System.out.println("Step size is " + stepSize);
			ArrayList<WritableComparable> result = new ArrayList<WritableComparable>(numPartitions-1);
			for(int i=1; i < numPartitions; i++) {
				result.add(records.get(Math.round(stepSize * i)));
			}
			
			return result;
		}
		
		 
	}
}
