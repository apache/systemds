package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.io.DataOutput;
import java.io.IOException;

/**
 * 
 * 
 */
public class CacheDataOutput implements DataOutput
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
		throw new IOException("Not supported.");
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
	private static final int BLOCK_NVALS = 512;
	private static final int BLOCK_NBYTES = BLOCK_NVALS*8;
	
	public void writeDoubleArray(int len, double[] varr) 
		throws IOException
	{
		if( _bufflen >=  BLOCK_NBYTES) //blockwise if buffer large enough
		{
			long tmp = -1;
			int i, j;
			
			//process full blocks of BLOCK_NVALS values 
			for( i=0; i<len-BLOCK_NVALS; i+=BLOCK_NVALS )
			{
				for( j=0; j<BLOCK_NVALS; j++ )
				{
					tmp = Double.doubleToLongBits(varr[i+j]);
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
			
			//process remaining values of the last block
			//(not relevant for performance, since at most BLOCK_NVALS-1 values)
			for(  ; i<len; i++ )
				writeDouble(varr[i]);
		}
		else //value wise (general case for small buffers)
		{
			for( int i=0; i<len; i++ )
				writeDouble(varr[i]);
		}
	}
}
