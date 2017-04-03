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

import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;
import org.apache.sysml.runtime.io.IOUtilFunctions;

public class CompactOutputFormat<K extends Writable, V extends Writable> extends FileOutputFormat<K,V> 
{

	
	static final String FINAL_SYNC_ATTRIBUTE = "final.sync";

	  /**
	   * Does the user want a final sync at close?
	   * 
	   * @param conf job configuration
	   * @return true if final sync at close
	   */
	  public static boolean getFinalSync(JobConf conf) {
	    return conf.getBoolean(FINAL_SYNC_ATTRIBUTE, false);
	  }
	
	public RecordWriter<K,V> getRecordWriter(FileSystem ignored, JobConf job, String name, Progressable progress) 
	throws IOException {
		
		Path file = FileOutputFormat.getTaskOutputPath(job, name);
		FileSystem fs = file.getFileSystem(job);
		FSDataOutputStream fileOut = fs.create(file, progress);
		return new FixedLengthRecordWriter<K,V>(fileOut, job);
	}
	
	public static class FixedLengthRecordWriter<K extends Writable, V extends Writable> implements RecordWriter<K, V> {

		private DataOutputStream out;
		 private boolean finalSync = false;
		
		public FixedLengthRecordWriter(DataOutputStream out, JobConf conf) {
			this.out = out;
			 finalSync = getFinalSync(conf);
		}
		
		@Override
		@SuppressWarnings("deprecation")
		public void close(Reporter reporter) throws IOException {
			if (finalSync) {
		        ((FSDataOutputStream) out).sync();
		      }
			IOUtilFunctions.closeSilently(out);
		}

		@Override
		public void write(K key, V value) throws IOException {
			key.write(out);
			value.write(out);
		}	
		
	}
	
}
