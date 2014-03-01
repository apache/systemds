/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.io.DataOutput;
import java.io.IOException;


import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSMDataOutput;
import com.ibm.bi.dml.runtime.matrix.io.SparseRow;

/**
 * Customer DataOutput to serialize directly into the given byte array.
 * 
 * 
 */
public class CacheDataOutput implements DataOutput, MatrixBlockDSMDataOutput 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	public void write(byte b[], int off, int len) 
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
		_buff[_count+0] = (byte)((v >>> 24) & 0xFF);
		_buff[_count+1] = (byte)((v >>> 16) & 0xFF);
		_buff[_count+2] = (byte)((v >>>  8) & 0xFF);
		_buff[_count+3] = (byte)((v >>>  0) & 0xFF);
		_count += 4;
	}
	
	@Override
	public void writeDouble(double v) 
		throws IOException 
	{
		long tmp = Double.doubleToRawLongBits(v);		
		_buff[_count+0] = (byte)((tmp >>> 56) & 0xFF);
		_buff[_count+1] = (byte)((tmp >>> 48) & 0xFF);
		_buff[_count+2] = (byte)((tmp >>> 40) & 0xFF);
		_buff[_count+3] = (byte)((tmp >>> 32) & 0xFF);
		_buff[_count+4] = (byte)((tmp >>> 24) & 0xFF);
		_buff[_count+5] = (byte)((tmp >>> 16) & 0xFF);
		_buff[_count+6] = (byte)((tmp >>>  8) & 0xFF);
		_buff[_count+7] = (byte)((tmp >>>  0) & 0xFF);		
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
		throw new IOException("Not supported.");
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
		for( int i=0; i<len; i++ )
		{
		    long tmp = Double.doubleToRawLongBits(varr[i]);
		    _buff[_count+0] = (byte)((tmp >>> 56) & 0xFF);
			_buff[_count+1] = (byte)((tmp >>> 48) & 0xFF);
			_buff[_count+2] = (byte)((tmp >>> 40) & 0xFF);
			_buff[_count+3] = (byte)((tmp >>> 32) & 0xFF);
			_buff[_count+4] = (byte)((tmp >>> 24) & 0xFF);
			_buff[_count+5] = (byte)((tmp >>> 16) & 0xFF);
			_buff[_count+6] = (byte)((tmp >>>  8) & 0xFF);
			_buff[_count+7] = (byte)((tmp >>>  0) & 0xFF);	
			_count+=8;
		}
	}
	
	@Override
	public void writeSparseRows(int rlen, SparseRow[] rows) 
		throws IOException
	{
		int lrlen = Math.min(rows.length, rlen);
		int i; //used for two consecutive loops
		
		//process existing rows
		for( i=0; i<lrlen; i++ )
		{
			SparseRow arow = rows[i];
			if( arow!=null && arow.size()>0 )
			{
				int alen = arow.size();
				int[] aix = arow.getIndexContainer();
				double[] avals = arow.getValueContainer();
				
				writeInt( alen );

				for( int j=0; j<alen; j++ )
				{
					int tmp1 = aix[j];
					_buff[_count+0 ] = (byte)((tmp1 >>> 24) & 0xFF);
					_buff[_count+1 ] = (byte)((tmp1 >>> 16) & 0xFF);
					_buff[_count+2 ] = (byte)((tmp1 >>>  8) & 0xFF);
					_buff[_count+3 ] = (byte)((tmp1 >>>  0) & 0xFF);
					
					long tmp2 = Double.doubleToRawLongBits(avals[j]);
					_buff[_count+4 ] = (byte)((tmp2 >>> 56) & 0xFF);
					_buff[_count+5 ] = (byte)((tmp2 >>> 48) & 0xFF);
					_buff[_count+6 ] = (byte)((tmp2 >>> 40) & 0xFF);
					_buff[_count+7 ] = (byte)((tmp2 >>> 32) & 0xFF);
					_buff[_count+8 ] = (byte)((tmp2 >>> 24) & 0xFF);
					_buff[_count+9 ] = (byte)((tmp2 >>> 16) & 0xFF);
					_buff[_count+10] = (byte)((tmp2 >>>  8) & 0xFF);
					_buff[_count+11] = (byte)((tmp2 >>>  0) & 0xFF);
					
					_count +=12;
				}	
			}
			else 
				writeInt( 0 );
		}
		
		//process remaining empty rows
		for( ; i<rlen; i++ )
			writeInt( 0 );
	}
}
