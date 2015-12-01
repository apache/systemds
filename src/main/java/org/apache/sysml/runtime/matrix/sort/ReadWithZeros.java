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


package com.ibm.bi.dml.runtime.matrix.sort;

import java.io.EOFException;
import java.io.IOException;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

public class ReadWithZeros 
{

	
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
