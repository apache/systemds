/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.io.DataInput;
import java.io.EOFException;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlockDataInput;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;

/**
 * 
 * 
 */
public class FastBufferedDataInputStream extends FilterInputStream implements DataInput, MatrixBlockDataInput
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected byte[] _buff;
	protected int _bufflen;
	
	public FastBufferedDataInputStream( InputStream in )
	{
		this(in, 8192);
	}
	
	public FastBufferedDataInputStream( InputStream in, int size )
	{
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
	public void readFully(byte[] b) 
		throws IOException 
	{
		readFully(b, 0, b.length);
	}

	@Override
	public void readFully(byte[] b, int off, int len)
		throws IOException 
	{
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
	public boolean readBoolean() 
		throws IOException 
	{
		return ( in.read() != 0 );
	}

	@Override
	public byte readByte()
		throws IOException 
	{
		return (byte)in.read();
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
		readFully(_buff, 0, 4);
		
		return baToInt(_buff, 0);
	}

	@Override
	public long readLong() 
		throws IOException 
	{
		readFully(_buff, 0, 8);
		
		return baToLong(_buff, 0);
	}

	@Override
	public float readFloat() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public double readDouble() 
		throws IOException 
	{
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
		throw new IOException("Not supported.");
	}

	
	///////////////////////////////////////////////
	// Implementation of MatrixBlockDataInput
	///////////////////////////////////////////////	

	@Override
	public int readDoubleArray(int len, double[] varr) 
		throws IOException 
	{
		//if( len<=0 || len != varr.length )
		//	throw new IndexOutOfBoundsException("len="+len+", varr.length="+varr.length);
		
		//counter for non-zero elements
		int nnz = 0;
		
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
	public int readSparseRows(int rlen, SparseRow[] rows) 
		throws IOException 
	{
		//counter for non-zero elements
		int nnz = 0;
		
		//read all individual sparse rows from input
		for( int i=0; i<rlen; i++ )
		{
			int lnnz = readInt();
			
			if( lnnz > 0 ) //non-zero row
			{
				//get handle to sparse (allocate if necessary)
				if( rows[i] == null )
					rows[i] = new SparseRow(lnnz);
				SparseRow arow = rows[i];
				
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
						arow.append(aix, aval);
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
						arow.append(aix, aval);
					}
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
