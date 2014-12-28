/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.EOFException;
import java.io.IOException;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

public class ReadWithZeros 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean contain0s=false;
	private long numZeros=0;
	private FSDataInputStream currentStream;
	
	private DoubleWritable keyAfterZero=new DoubleWritable();
	private IntWritable valueAfterZero=new IntWritable(); 
	private boolean justFound0=false;
	
	public ReadWithZeros(FSDataInputStream in, boolean contain0, long num0)
	{
		currentStream=in;
		contain0s=contain0;
		numZeros=num0;
	}
	
	public void readNextKeyValuePairs(DoubleWritable readKey, IntWritable readValue)throws IOException 
	{
		try {
			if(contain0s && justFound0)
			{
				readKey.set(keyAfterZero.get());
				readValue.set(valueAfterZero.get());
				contain0s=false;
			}else
			{
				readKey.readFields(currentStream);
				readValue.readFields(currentStream);
			}
		} catch(EOFException e) {
			// case in which zero is the maximum value in the matrix. 
			// The zero value from the last entry is not present in the input sorted matrix, but needs to be accounted for.
			if (contain0s && !justFound0 ) {
				justFound0=true;
				readKey.set(0);
				readValue.set((int)numZeros);
			}
			else {
				throw e;
			}
		}
		
		if(contain0s && !justFound0 && readKey.get()>=0)
		{
			justFound0=true;
			keyAfterZero.set(readKey.get());
			valueAfterZero.set(readValue.get());
			readKey.set(0);
			readValue.set((int)numZeros);
		}
	}
}
