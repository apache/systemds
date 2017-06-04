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

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.sysml.runtime.io.IOUtilFunctions;

@SuppressWarnings("rawtypes")
public class CompactInputFormat<K extends WritableComparable, V extends Writable> extends FileInputFormat<K, V>  
{

	
	public static final String KEY_CLASS="compact.fixed.length.input.key.class";
	public static final String VALUE_CLASS="compact.fixed.length.input.value.class";
	
	public static void setKeyValueClasses(JobConf job, Class<? extends WritableComparable> keyClass, Class<? extends Writable> valueClass)
	{
		job.setClass(KEY_CLASS, keyClass, WritableComparable.class);
		job.setClass(VALUE_CLASS, valueClass, Writable.class);
	}
	
	public RecordReader<K,V> getRecordReader(InputSplit split
			, JobConf job, Reporter reporter) throws IOException {
		return new CompactInputRecordReader<K,V>(job, (FileSplit) split);
	}
	
	//the files are not splitable
	protected boolean isSplitable(FileSystem fs, Path filename)
	{
		return false;
	}
	
	public static class CompactInputRecordReader<K extends WritableComparable, V extends Writable> 
	implements RecordReader<K, V>
	{

		protected long totLength;
		protected FileSystem fs;
		protected Path path;
		protected Class<? extends WritableComparable> keyClass;
		protected Class<? extends Writable> valueClass;
		protected JobConf conf;
	    
	    protected FSDataInputStream currentStream;
	    
		@SuppressWarnings("unchecked")
		public CompactInputRecordReader(JobConf job, FileSplit split) throws IOException {

	    	path = split.getPath();
	    	fs = IOUtilFunctions.getFileSystem(path, job);
			totLength = split.getLength();
	    	currentStream = fs.open(path);
	    	keyClass=(Class<? extends WritableComparable>) job.getClass(KEY_CLASS, WritableComparable.class);
	    	valueClass=(Class<? extends Writable>) job.getClass(VALUE_CLASS, Writable.class);
	    }
	    
		@Override
		public void close() throws IOException {
			IOUtilFunctions.closeSilently(currentStream);
		}
		@SuppressWarnings("unchecked")
		public K createKey() {
			return (K) ReflectionUtils.newInstance(keyClass, conf);
		}
	  
		@SuppressWarnings("unchecked")
		public V createValue() {
			return (V) ReflectionUtils.newInstance(valueClass, conf);
		}

		@Override
		public long getPos() throws IOException {
			long currentOffset = currentStream == null ? 0 : currentStream.getPos();
			return currentOffset;
		}

		@Override
		public float getProgress() throws IOException {
			return totLength==0 ? 0 : ((float)getPos()) / totLength;
		}

		@Override
		public boolean next(K key, V value)
				throws IOException {
		    if(currentStream.getPos()<totLength)
		    {
		    	key.readFields(currentStream);
				value.readFields(currentStream);
				return true;
		    }else
		    	return false;
		}
	}
}
