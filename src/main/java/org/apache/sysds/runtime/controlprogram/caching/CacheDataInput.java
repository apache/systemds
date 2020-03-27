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

package org.apache.sysds.runtime.controlprogram.caching;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.IOException;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlockDataInput;

public class CacheDataInput implements DataInput, MatrixBlockDataInput
{
	protected final byte[] _buff;
	protected int _count;

	public CacheDataInput(byte[] mem) {
		_buff = mem;
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
	public boolean readBoolean() throws IOException {
		//mask to adhere to the input stream semantic
		return ( (_buff[_count++] & 0xFF) != 0 );
	}

	@Override
	public byte readByte() throws IOException {
		//mask to adhere to the input stream semantic
		return (byte) (_buff[_count++] & 0xFF);
	}

	@Override
	public int readUnsignedByte() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public short readShort() throws IOException {
		int ret = baToShort(_buff, _count);
		_count += 2;
		return (short) ret;
	}

	@Override
	public int readUnsignedShort() throws IOException {
		int ret = baToShort(_buff, _count);
		_count += 2;
		return ret;
	}

	@Override
	public char readChar() throws IOException {
		int ret = baToShort(_buff, _count);
		_count += 2;
		return (char) ret;
	}

	@Override
	public int readInt() throws IOException {
		int ret = baToInt(_buff, _count);
		_count += 4;
		return ret;
	}

	@Override
	public long readLong() throws IOException {
		long ret = baToLong(_buff, _count);
		_count += 8;
		return ret;
	}

	@Override
	public float readFloat() throws IOException {
		int tmp = baToInt(_buff, _count);
		float tmp2 = Float.intBitsToFloat(tmp);
		_count += 4;
		return tmp2;
	}

	@Override
	public double readDouble() throws IOException {
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
		return DataInputStream.readUTF(this);
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
	public long readSparseRows(int rlen, long nnz, SparseBlock rows) 
		throws IOException 
	{
		//counter for non-zero elements
		long gnnz = 0;
		
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
				
				gnnz += lnnz;	
			}
		}
		
		//sanity check valid number of read nnz
		if( gnnz != nnz )
			throw new IOException("Invalid number of read nnz: "+gnnz+" vs "+nnz);
		
		return nnz;
	}

	private static int baToShort( byte[] ba, final int off ) {
		return IOUtilFunctions.baToShort(ba, off);
	}

	private static int baToInt( byte[] ba, final int off ) {
		return IOUtilFunctions.baToInt(ba, off);
	}

	private static long baToLong( byte[] ba, final int off ) {
		return IOUtilFunctions.baToLong(ba, off);
	}
}
