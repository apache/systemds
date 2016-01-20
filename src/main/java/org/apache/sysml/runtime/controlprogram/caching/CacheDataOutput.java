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

import java.io.DataOutput;
import java.io.IOException;

import org.apache.sysml.runtime.matrix.data.MatrixBlockDataOutput;
import org.apache.sysml.runtime.matrix.data.SparseBlock;

/**
 * Customer DataOutput to serialize directly into the given byte array.
 * 
 * 
 */
public class CacheDataOutput implements DataOutput, MatrixBlockDataOutput 
{
	
	protected byte[] _buff;
	protected int _bufflen;
	protected int _count;

	public CacheDataOutput( byte[] mem ) 
	{		
		_buff = mem;
		_bufflen = _buff.length;
		_count = 0;
	}
	
	@Override
	public void write(int b) 
    	throws IOException 
    {
		_buff[_count++] = (byte)b;
    }

    @Override
	public void write(byte[] b) 
		throws IOException 
	{
		System.arraycopy(b, 0, _buff, _count, b.length);
		_count += b.length;
	}
    
    @Override
	public void write(byte[] b, int off, int len) 
    	throws IOException 
    {
		System.arraycopy(b, off, _buff, _count, len);
		_count += len;
    }
	
	@Override
	public void writeBoolean(boolean v) 
		throws IOException 
	{
		_buff[_count++] = (byte)( v ? 1 : 0 );
	}


	@Override
	public void writeInt(int v) 
		throws IOException 
	{
		intToBa(v, _buff, _count);
		_count += 4;
	}
	
	@Override
	public void writeDouble(double v) 
		throws IOException 
	{
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
		throw new IOException("Not supported.");
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
	public void writeLong(long v) throws IOException {
		longToBa(v,  _buff, _count);
		_count += 8;
	}

	@Override
	public void writeShort(int v) throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public void writeUTF(String s) throws IOException {
		throw new IOException("Not supported.");
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
		for( int i=0; i<len; i++ )
		{
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
	
	/**
	 * 
	 * @param val
	 * @param ba
	 * @param off
	 */
	private static void intToBa( final int val, byte[] ba, final int off )
	{
		//shift and mask out 4 bytes
		ba[ off+0 ] = (byte)((val >>> 24) & 0xFF);
		ba[ off+1 ] = (byte)((val >>> 16) & 0xFF);
		ba[ off+2 ] = (byte)((val >>>  8) & 0xFF);
		ba[ off+3 ] = (byte)((val >>>  0) & 0xFF);
	}
	
	/**
	 * 
	 * @param val
	 * @param ba
	 * @param off
	 */
	private static void longToBa( final long val, byte[] ba, final int off )
	{
		//shift and mask out 8 bytes
		ba[ off+0 ] = (byte)((val >>> 56) & 0xFF);
		ba[ off+1 ] = (byte)((val >>> 48) & 0xFF);
		ba[ off+2 ] = (byte)((val >>> 40) & 0xFF);
		ba[ off+3 ] = (byte)((val >>> 32) & 0xFF);
		ba[ off+4 ] = (byte)((val >>> 24) & 0xFF);
		ba[ off+5 ] = (byte)((val >>> 16) & 0xFF);
		ba[ off+6 ] = (byte)((val >>>  8) & 0xFF);
		ba[ off+7 ] = (byte)((val >>>  0) & 0xFF);
	}
}
