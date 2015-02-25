/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

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

public class UnPaddedOutputFormat<K extends Writable, V extends Writable> extends FileOutputFormat<K, V>
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static class UnpaddedRecordWriter<K extends Writable, V extends Writable> implements RecordWriter<K, V>
	{
		/** file input stream */
		private FSDataOutputStream out;

		public UnpaddedRecordWriter(FSDataOutputStream fstream)
		{
			out=fstream;
		}
		@Override
		public void close(Reporter report) throws IOException {
			out.close();
			
		}
		@Override
		public void write(K key, V value) throws IOException {
			key.write(out);
			value.write(out);
		}
	}

	@SuppressWarnings("deprecation")
	@Override
	public RecordWriter<K, V> getRecordWriter(FileSystem ignored, JobConf job,
			String name, Progressable progress) throws IOException {
		Path file = FileOutputFormat.getTaskOutputPath(job, name);
	    FileSystem fs = file.getFileSystem(job);
	    FSDataOutputStream fileOut = fs.create(file, true, job.getInt("io.file.buffer.size", 4096), progress);
		return new UnpaddedRecordWriter<K, V>(fileOut);
	}
}