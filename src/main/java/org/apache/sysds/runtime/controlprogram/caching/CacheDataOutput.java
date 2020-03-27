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

import java.io.DataOutput;
import java.io.IOException;
import java.io.UTFDataFormatException;

import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlockDataOutput;

/**
 * Custom DataOutput to serialize directly into the given byte array.
 * 
 */
public class CacheDataOutput implements DataOutput, MatrixBlockDataOutput 
{
	protected final byte[] _buff;
	protected int _count;

	public CacheDataOutput(int size) {
		this(new byte[size]);
	}
	
	public CacheDataOutput(byte[] mem) {
		_buff = mem;
		_count = 0;
	}
	
	@Override
	public void write(int b) throws IOException {
		_buff[_count++] = (byte)b;
	}

	@Override
	public void write(byte[] b) throws IOException {
		System.arraycopy(b, 0, _buff, _count, b.length);
		_count += b.length;
	}

	@Override
	public void write(byte[] b, int off, int len) throws IOException {
		System.arraycopy(b, off, _buff, _count, len);
		_count += len;
	}
	
	@Override
	public void writeBoolean(boolean v) throws IOException {
		_buff[_count++] = (byte)( v ? 1 : 0 );
	}

	@Override
	public void writeInt(int v) throws IOException {
		intToBa(v, _buff, _count);
		_count += 4;
	}
	
	@Override
	public void writeDouble(double v) throws IOException {
		long tmp = Double.doubleToRawLongBits(v);
		longToBa(tmp, _buff, _count);
		_count += 8;
	}

	@Override
	public void writeByte(int v) throws IOException {
		_buff[_count++] = (byte) v;
	}

	@Override
	public void writeBytes(String s) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public void writeChar(int v) throws IOException {
		writeShort(v);
	}

	@Override
	public void writeChars(String s) throws IOException {
		throw new IOException("Not supported.");
	}
	
	@Override
	public void writeFloat(float v) throws IOException {
		int tmp = Float.floatToRawIntBits(v);
		intToBa(tmp, _buff, _count);
		_count += 4;
	}

	@Override
	public void writeLong(long v) throws IOException {
		longToBa(v, _buff, _count);
		_count += 8;
	}

	@Override
	public void writeShort(int v) throws IOException {
		shortToBa(v, _buff, _count);
		_count += 2;
	}

	@Override
	public void writeUTF(String s) throws IOException {
		int slen = s.length();
		int utflen = IOUtilFunctions.getUTFSize(s) - 2;
		if (utflen-2 > 65535)
			throw new UTFDataFormatException("encoded string too long: "+utflen);
		
		//write utf len (2 bytes) 
		writeShort(utflen);
		
		//write utf payload
		for( int i=0; i<slen; i++ ) {
			char c = s.charAt(i);
			if( c>= 0x0001 && c<=0x007F ) //1 byte range
				writeByte(c);
			else if( c>=0x0800 ) { //3 byte range
				_buff[_count++] = (byte) (0xE0 | ((c >> 12) & 0x0F));
				_buff[_count++] = (byte) (0x80 | ((c >>  6) & 0x3F));
				_buff[_count++] = (byte) (0x80 | ((c >>  0) & 0x3F));
			}
			else { //2 byte range and null
				_buff[_count++] = (byte) (0xC0 | ((c >>  6) & 0x1F));
				_buff[_count++] = (byte) (0x80 | ((c >>  0) & 0x3F));
			}
		}
	}

	///////////////////////////////////////////////
	// Implementation of MatrixBlockDSMDataOutput
	///////////////////////////////////////////////

	@Override
	public void writeDoubleArray(int len, double[] varr) 
		throws IOException
	{
		//original buffer offset
		int off = _count;
		
		//serialize entire array into buffer
		for( int i=0; i<len; i++ ) {
			long tmp = Double.doubleToRawLongBits(varr[i]);
			longToBa(tmp, _buff, off+i*8);
		}
		
		//update buffer offset
		_count = off + len*8;
	}
	
	@Override
	public void writeSparseRows(int rlen, SparseBlock rows) 
		throws IOException
	{
		int lrlen = Math.min(rows.numRows(), rlen);
		
		//process existing rows
		for( int i=0; i<lrlen; i++ )
		{
			if( !rows.isEmpty(i) )
			{
				int apos = rows.pos(i);
				int alen = rows.size(i);
				int[] aix = rows.indexes(i);
				double[] avals = rows.values(i);
				
				writeInt( alen );

				for( int j=apos; j<apos+alen; j++ )
				{
					intToBa(aix[j], _buff, _count);
					long tmp2 = Double.doubleToRawLongBits(avals[j]);
					longToBa(tmp2, _buff, _count+4);
					_count += 12;
				}	
			}
			else 
				writeInt( 0 );
		}
		
		//process remaining empty rows
		for( int i=lrlen; i<rlen; i++ )
			writeInt( 0 );
	}
	
	public byte[] getBytes() {
		return _buff;
	}

	private static void shortToBa( final int val, byte[] ba, final int off ) {
		IOUtilFunctions.shortToBa(val, ba, off);
	}

	private static void intToBa( final int val, byte[] ba, final int off ) {
		IOUtilFunctions.intToBa(val, ba, off);
	}

	private static void longToBa( final long val, byte[] ba, final int off ) {
		IOUtilFunctions.longToBa(val, ba, off);
	}
}
