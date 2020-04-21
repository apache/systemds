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

package org.apache.sysds.runtime.util;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlockDataInput;

public class FastBufferedDataInputStream extends FilterInputStream implements DataInput, MatrixBlockDataInput
{
	protected byte[] _buff;
	protected int _bufflen;
	
	public FastBufferedDataInputStream( InputStream in ) {
		this(in, 8192);
	}
	
	public FastBufferedDataInputStream( InputStream in, int size ) {
		super(in);
		if (size <= 0) 
			throw new IllegalArgumentException("Buffer size <= 0");
		_buff = new byte[ size ];
		_bufflen = size;
	}

	/////////////////////////////
	// DataInput Implementation
	/////////////////////////////

	@Override
	public void readFully(byte[] b) throws IOException {
		readFully(b, 0, b.length);
	}

	@Override
	public void readFully(byte[] b, int off, int len) throws IOException {
		if (len < 0)
			throw new IndexOutOfBoundsException();
		int n = 0;
		while (n < len) {
			int count = in.read(b, off + n, len - n);
			if (count < 0)
				throw new EOFException();
			n += count;
		}
	}

	@Override
	public int skipBytes(int n) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public boolean readBoolean() throws IOException {
		return in.read() != 0;
	}

	@Override
	public byte readByte() throws IOException {
		return (byte)in.read();
	}

	@Override
	public int readUnsignedByte() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public short readShort() throws IOException {
		readFully(_buff, 0, 2);
		return (short) baToShort(_buff, 0);
	}

	@Override
	public int readUnsignedShort() throws IOException {
		readFully(_buff, 0, 2);
		return baToShort(_buff, 0);
	}

	@Override
	public char readChar() throws IOException {
		readFully(_buff, 0, 2);
		return (char)baToShort(_buff, 0);
	}

	@Override
	public int readInt() throws IOException {
		readFully(_buff, 0, 4);
		return baToInt(_buff, 0);
	}

	@Override
	public long readLong() throws IOException {
		readFully(_buff, 0, 8);
		return baToLong(_buff, 0);
	}

	@Override
	public float readFloat() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public double readDouble() throws IOException {
		readFully(_buff, 0, 8);
		long tmp = baToLong(_buff, 0);
		return Double.longBitsToDouble( tmp );
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
	// Implementation of MatrixBlockDataInput
	///////////////////////////////////////////////

	@Override
	public long readDoubleArray(int len, double[] varr) 
		throws IOException 
	{
		if( len<=0 || len > varr.length )
			throw new IndexOutOfBoundsException("len="+len+", varr.length="+varr.length);
		
		//counter for non-zero elements
		long nnz = 0;
		
		//outer loop for buffered read
		for( int i=0; i<len; i+=_bufflen/8 ) 
		{
			//read next 8KB block from input 
			//note: cast to long to prevent overflows w/ len*8
			int maxNB = (int)Math.min(_bufflen, ((long)len-i)*8);
			readFully(_buff, 0, maxNB);
			
			for( int j=0, ix=i; j<maxNB; j+=8, ix++ ) 
			{
				//core deserialization
				long tmp = baToLong(_buff, j);
				varr[ix] = Double.longBitsToDouble( tmp );
				
				//nnz maintenance
				nnz += (varr[ix]!=0) ? 1 : 0; 
			}
		}
		
		return nnz;
	}

	@Override
	public long readSparseRows(int rlen, long nnz, SparseBlock rows) 
		throws IOException 
	{
		//check for CSR quick-path
		if( rows instanceof SparseBlockCSR ) {
			((SparseBlockCSR) rows).initSparse(rlen, (int)nnz, this);
			return nnz;
		}
		
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
				//note: cast to long to prevent overflows w/ lnnz*12
				if( ((long)lnnz*12) < _bufflen )
				{
					//single buffer read if sparse row fits in buffer
					readFully(_buff, 0, lnnz*12);
								
					for( int j=0; j<lnnz*12; j+=12 ) 
					{	
						int aix = baToInt(_buff, j);
						long tmp = baToLong(_buff, j+4);
						double aval = Double.longBitsToDouble( tmp );
						rows.append(i, aix, aval);
					}
				}
				else
				{
					//default case: by value-pair
					for( int j=0; j<lnnz; j++ ) {
						readFully(_buff, 0, 12);
						int aix = baToInt(_buff, 0);
						long tmp = baToLong(_buff, 4);
						double aval = Double.longBitsToDouble(tmp);
						rows.append(i, aix, aval);
					}
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
