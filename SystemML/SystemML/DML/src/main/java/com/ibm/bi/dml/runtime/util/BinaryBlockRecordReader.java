/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.SequenceFileRecordReader;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;

/**
 * Custom record reader for binary block. Currently its only purpose is to allow for
 * detailed profiling of overall read time (io, deserialize, decompress).
 * 
 * NOTE: not used by default.
 */
public class BinaryBlockRecordReader extends SequenceFileRecordReader<MatrixIndexes,MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private long _time = 0;
	
	public BinaryBlockRecordReader(Configuration conf, FileSplit split)
		throws IOException 
	{
		super(conf, split);
		
	}
	
	@Override
	public synchronized boolean next(MatrixIndexes key, MatrixBlock value)
		throws IOException 
	{
		long t0 = System.nanoTime();		
		boolean ret = super.next(key, value);		
		long t1 = System.nanoTime();
		
		_time+=(t1-t0);
		
		return ret;
	}

	@Override
	public synchronized void close() 
		throws IOException 
	{		
		//in milliseconds.
		//System.out.println(_time/1000000);
		super.close();
	}
}
