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

package org.tugraz.sysds.runtime.util;

import java.io.DataOutput;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UTFDataFormatException;

import org.tugraz.sysds.runtime.data.SparseBlock;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlockDataOutput;

/**
 * This buffered output stream is essentially a merged version of
 * BufferedOutputStream and DataOutputStream, wrt SystemDS requirements.
 * 
 * Micro-benchmarks showed a 25% performance improvement for local write binary block
 * due to the following advantages: 
 * - 1) unsynchronized buffered output stream (not required in SystemDS since single writer)
 * - 2) single output buffer (avoid two-level buffers of individual streams)
 * - 3) specific support for writing double arrays in a blockwise fashion
 * 
 */
public class FastBufferedDataOutputStream extends FilterOutputStream implements DataOutput, MatrixBlockDataOutput
{
	protected byte[] _buff;
	protected int _bufflen;
	protected int _count;

	public FastBufferedDataOutputStream(OutputStream out) {
		this(out, 8192);
	}

	public FastBufferedDataOutputStream(OutputStream out, int size) {
		super(out);
		if(size <= 0)
			throw new IllegalArgumentException("Buffer size <= 0.");
		if( size%8 != 0 )
			throw new IllegalArgumentException("Buffer size not a multiple of 8.");
		_buff = new byte[size];
		_bufflen = size;
	}

	@Override
	public void write(int b) throws IOException {
		if (_count >= _bufflen)
			flushBuffer();
		_buff[_count++] = (byte)b;
	}

	@Override
	public void write(byte[] b, int off, int len) 
		throws IOException 
	{
		if (len >= _bufflen) {
			flushBuffer();
			out.write(b, off, len);
			return;
		}
		if (len > _bufflen - _count) {
			flushBuffer();
		}
		System.arraycopy(b, off, _buff, _count, len);
		_count += len;
	}

	@Override
	public void flush() throws IOException {
		flushBuffer();
		out.flush();
	}

	private void flushBuffer() throws IOException {
		if(_count > 0) {
			out.write(_buff, 0, _count);
			_count = 0;
		}
	}

	@Override
	public void close() throws IOException {
		super.close();
	}

	/////////////////////////////
	// DataOutput Implementation
	/////////////////////////////

	@Override
	public void writeBoolean(boolean v) throws IOException  {
		if (_count >= _bufflen)
			flushBuffer();
		_buff[_count++] = (byte)(v ? 1 : 0);
	}


	@Override
	public void writeInt(int v) throws IOException {
		if (_count+4 > _bufflen)
			flushBuffer();
		intToBa(v, _buff, _count);
		_count += 4;
	}

	@Override
	public void writeLong(long v) throws IOException {
		if (_count+8 > _bufflen)
			flushBuffer();
		longToBa(v, _buff, _count);
		_count += 8;
	}
	
	@Override
	public void writeDouble(double v) throws IOException {
		if (_count+8 > _bufflen)
			flushBuffer();
		long tmp = Double.doubleToRawLongBits(v);
		longToBa(tmp, _buff, _count);
		_count += 8;
	}

	@Override
	public void writeByte(int v) throws IOException {
		if (_count+1 > _bufflen)
			flushBuffer();
		_buff[_count++] = (byte) v;
	}

	@Override
	public void writeShort(int v) throws IOException {
		if (_count+2 > _bufflen)
			flushBuffer();
		shortToBa(v, _buff, _count);
		_count += 2;
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
		throw new IOException("Not supported.");
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
			if (_count+3 > _bufflen)
				flushBuffer();
			char c = s.charAt(i);
			if( c>= 0x0001 && c<=0x007F ) //1 byte range
				_buff[_count++] = (byte) c;
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
		//initial flush
		flushBuffer();
		
		//write matrix block-wise to underlying stream
		//(increase i in awareness of len to prevent int overflow)
		int blen = _bufflen/8;
		for( int i=0; i<len; i+=Math.min(len-i, blen) )
		{
			//write values of current block
			int lblen = Math.min(len-i, blen);
			for( int j=0; j<lblen; j++ )
			{
				long tmp = Double.doubleToRawLongBits(varr[i+j]);
				longToBa(tmp, _buff, _count);
				_count += 8;
			}	
			
			//flush buffer for current block
			flushBuffer(); //based on count
		}
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
				int alen2 = alen*12;
				int[] aix = rows.indexes(i);
				double[] avals = rows.values(i);
				
				writeInt( alen );
				
				if( alen2 < _bufflen )
				{
					if (_count+alen2 > _bufflen) 
					    flushBuffer();
					
					for( int j=apos; j<apos+alen; j++ )
					{
						long tmp2 = Double.doubleToRawLongBits(avals[j]);
						intToBa(aix[j], _buff, _count);
						longToBa(tmp2, _buff, _count+4);
						_count += 12;
					}
				}
				else
				{
					//row does not fit in buffer
					for( int j=apos; j<apos+alen; j++ )
					{
						if (_count+12 > _bufflen) 
						    flushBuffer();
						
						long tmp2 = Double.doubleToRawLongBits(avals[j]);
						intToBa(aix[j], _buff, _count);
						longToBa(tmp2, _buff, _count+4);
						_count += 12;
					}
				}	
			}
			else 
				writeInt( 0 );
		}
		
		//process remaining empty rows
		for( int i=lrlen; i<rlen; i++ )
			writeInt( 0 );
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
