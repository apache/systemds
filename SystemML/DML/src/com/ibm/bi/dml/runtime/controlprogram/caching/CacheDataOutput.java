/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.io.DataOutput;
import java.io.IOException;

/**
 * 
 * 
 */
public class CacheDataOutput implements DataOutput
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
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
		_buff[_count++] = (byte)(v?1:0);
	}


	@Override
	public void writeInt(int v) 
		throws IOException 
	{
		_buff[_count++] = (byte)((v >>> 24) & 0xFF);
		_buff[_count++] = (byte)((v >>> 16) & 0xFF);
		_buff[_count++] = (byte)((v >>>  8) & 0xFF);
		_buff[_count++] = (byte)((v       ) & 0xFF);
	}
	
	@Override
	public void writeDouble(double v) 
		throws IOException 
	{
		long tmp = Double.doubleToLongBits(v);		
		_buff[_count++] = (byte)((tmp >>> 56) & 0xFF);
		_buff[_count++] = (byte)((tmp >>> 48) & 0xFF);
		_buff[_count++] = (byte)((tmp >>> 40) & 0xFF);
		_buff[_count++] = (byte)((tmp >>> 32) & 0xFF);
		_buff[_count++] = (byte)((tmp >>> 24) & 0xFF);
		_buff[_count++] = (byte)((tmp >>> 16) & 0xFF);
		_buff[_count++] = (byte)((tmp >>>  8) & 0xFF);
		_buff[_count++] = (byte)((tmp       ) & 0xFF);		
	}

	@Override
	public void writeByte(int v) throws IOException {
		_buff[_count++] = (byte)((v) & 0xFF);	
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


    /////////////////////////////////////////
    // Custom implementation for arrays
    /////////////////////////////////////////	
	
	public void writeDoubleArray(int len, double[] varr) 
		throws IOException
	{
		long tmp = -1;
		
		for( int i=0; i<len; i++ )
		{
			tmp = Double.doubleToLongBits(varr[i]);
			_buff[_count++] = (byte)((tmp >>> 56) & 0xFF);
			_buff[_count++] = (byte)((tmp >>> 48) & 0xFF);
			_buff[_count++] = (byte)((tmp >>> 40) & 0xFF);
			_buff[_count++] = (byte)((tmp >>> 32) & 0xFF);
			_buff[_count++] = (byte)((tmp >>> 24) & 0xFF);
			_buff[_count++] = (byte)((tmp >>> 16) & 0xFF);
			_buff[_count++] = (byte)((tmp >>>  8) & 0xFF);
			_buff[_count++] = (byte)((tmp       ) & 0xFF);	
		}
	}
}
