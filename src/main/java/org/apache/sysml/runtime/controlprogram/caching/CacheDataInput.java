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

package org.apache.sysml.runtime.controlprogram.caching;

import java.io.DataInput;
import java.io.IOException;

import org.apache.sysml.runtime.matrix.data.MatrixBlockDataInput;
import org.apache.sysml.runtime.matrix.data.SparseBlock;

public class CacheDataInput implements DataInput, MatrixBlockDataInput
{
	
	protected byte[] _buff;
	protected int _bufflen;
	protected int _count;

	public CacheDataInput( byte[] mem ) 
	{		
		_buff = mem;
		_bufflen = _buff.length;
		_count = 0;
	}

	@Override
	public void readFully(byte[] b) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public void readFully(byte[] b, int off, int len) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public int skipBytes(int n) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public boolean readBoolean() 
		throws IOException 
	{
		//mask to adhere to the input stream semantic
		return ( (_buff[_count++] & 0xFF) != 0 );
	}

	@Override
	public byte readByte()
		throws IOException 
	{
		//mask to adhere to the input stream semantic
		return (byte) (_buff[_count++] & 0xFF);
	}

	@Override
	public int readUnsignedByte() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public short readShort() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public int readUnsignedShort() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public char readChar() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public int readInt() 
		throws IOException 
	{
		int ret = baToInt(_buff, _count);
		_count += 4;
		
		return ret;
	}

	@Override
	public long readLong() 
		throws IOException 
	{
		long ret = baToLong(_buff, _count);
		_count += 8;
		
		return ret;
	}

	@Override
	public float readFloat() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public double readDouble() 
		throws IOException 
	{
		long tmp = baToLong(_buff, _count);
		double tmp2 = Double.longBitsToDouble(tmp);
		_count += 8;
		
		return tmp2;
	}

	@Override
	public String readLine() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public String readUTF() throws IOException {
		throw new IOException("Not supported.");
	}
	
    ///////////////////////////////////////////////
    // Implementation of MatrixBlockDSMDataOutput
    ///////////////////////////////////////////////	
	
	@Override
	public long readDoubleArray(int len, double[] varr) 
		throws IOException 
	{
		//counter for non-zero elements
		long nnz = 0;
		
		int off = _count;
		for( int i=0; i<len; i++ ) 
		{
			//core deserialization
			long tmp = baToLong(_buff, off+i*8);
			varr[i] = Double.longBitsToDouble( tmp );
			
			//nnz maintenance
			nnz += (varr[i]!=0) ? 1 : 0; 
		}
		_count = off + len*8;
		
		return nnz;
	}

	@Override
	public long readSparseRows(int rlen, SparseBlock rows) 
		throws IOException 
	{
		//counter for non-zero elements
		long nnz = 0;
		
		//read all individual sparse rows from input
		for( int i=0; i<rlen; i++ )
		{
			int lnnz = readInt();
			
			if( lnnz > 0 ) //non-zero row
			{
				//get handle to sparse (allocate if necessary)
				rows.allocate(i, lnnz);
				
				//read single sparse row
				for( int j=0; j<lnnz; j++ ) 
				{	
					int aix = baToInt(_buff, _count);
					long tmp = baToLong(_buff, _count+4);
					double aval = Double.longBitsToDouble( tmp );
					rows.append(i, aix, aval);
					_count+=12;
				}
				
				nnz += lnnz;	
			}
		}
		
		return nnz;
	}
	
	/**
	 * 
	 * @param a
	 * @param off
	 * @return
	 */
	private static int baToInt( byte[] ba, final int off )
	{
		//shift and add 4 bytes into single int
		return ((ba[off+0] & 0xFF) << 24) +
			   ((ba[off+1] & 0xFF) << 16) +
			   ((ba[off+2] & 0xFF) <<  8) +
			   ((ba[off+3] & 0xFF) <<  0);
	}
	
	/**
	 * 
	 * @param a
	 * @param off
	 * @return
	 */
	private static long baToLong( byte[] ba, final int off )
	{
		//shift and add 8 bytes into single long
		return ((long)(ba[off+0] & 0xFF) << 56) +
               ((long)(ba[off+1] & 0xFF) << 48) +
	           ((long)(ba[off+2] & 0xFF) << 40) +
               ((long)(ba[off+3] & 0xFF) << 32) +
               ((long)(ba[off+4] & 0xFF) << 24) +
               ((long)(ba[off+5] & 0xFF) << 16) +
               ((long)(ba[off+6] & 0xFF) <<  8) +
               ((long)(ba[off+7] & 0xFF) <<  0);
	}
}
