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
import java.io.IOException;
import java.nio.ByteBuffer;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlockDataInput;

public class ByteBufferDataInput implements DataInput, MatrixBlockDataInput
{
	protected final ByteBuffer _buff;

	public ByteBufferDataInput(ByteBuffer buff) {
		_buff = buff;
	}

	public int available() {
		return _buff.limit() - _buff.position();
	}
	
	@Override
	public void readFully(byte[] b) throws IOException {
		_buff.get(b);
	}

	@Override
	public void readFully(byte[] b, int off, int len) throws IOException {
		_buff.get(b, off, len);
	}

	@Override
	public int skipBytes(int n) throws IOException {
		_buff.position(_buff.position()+n);
		return n;
	}

	@Override
	public boolean readBoolean() throws IOException {
		//mask to adhere to the input stream semantic
		return ( (_buff.get() & 0xFF) != 0 );
	}

	@Override
	public byte readByte() throws IOException {
		//mask to adhere to the input stream semantic
		return (byte) (_buff.get() & 0xFF);
	}

	@Override
	public int readUnsignedByte() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public short readShort() throws IOException {
		return _buff.getShort();
	}

	@Override
	public int readUnsignedShort() throws IOException {
		return _buff.getChar();
	}

	@Override
	public char readChar() throws IOException {
		return _buff.getChar();
	}

	@Override
	public int readInt() throws IOException {
		return _buff.getInt();
	}

	@Override
	public long readLong() throws IOException {
		return _buff.getLong();
	}

	@Override
	public float readFloat() throws IOException {
		return _buff.getFloat();
	}

	@Override
	public double readDouble() throws IOException {
		return _buff.getDouble();
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
	public long readDoubleArray(int len, double[] varr) throws IOException  {
		long nnz = 0;
		for( int i=0; i<len; i++ )
			nnz += (varr[i] = _buff.getDouble()) != 0 ? 1 : 0;
		return nnz;
	}

	@Override
	public long readSparseRows(int rlen, long nnz, SparseBlock rows) 
		throws IOException 
	{
		//counter for non-zero elements
		long gnnz = 0;
		
		//read all individual sparse rows from input
		for( int i=0; i<rlen; i++ ) {
			int lnnz = _buff.getInt();
			if( lnnz > 0 ) { //non-zero row
				rows.allocate(i, lnnz); //preallocate row
				for( int j=0; j<lnnz; j++ ) //read single sparse row
					rows.append(i, _buff.getInt(), _buff.getDouble());
				gnnz += lnnz;
			}
		}
		
		//sanity check valid number of read nnz
		if( gnnz != nnz )
			throw new IOException("Invalid number of read nnz: "+gnnz+" vs "+nnz);
		
		return nnz;
	}
}
