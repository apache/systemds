/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.io.DataInput;
import java.io.IOException;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlockDataInput;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;

public class CacheDataInput implements DataInput, MatrixBlockDataInput
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected byte[] _buff;
	protected int _bufflen;
	protected int _count;

	public CacheDataInput( byte[] mem ) 
	{		
		_buff = mem;
		_bufflen = _buff.length;
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
	public boolean readBoolean() 
		throws IOException 
	{
		//mask to adhere to the input stream semantic
		return ( (_buff[_count++] & 0xFF) != 0 );
	}

	@Override
	public byte readByte()
		throws IOException 
	{
		//mask to adhere to the input stream semantic
		return (byte) (_buff[_count++] & 0xFF);
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
		int ret = baToInt(_buff, _count);
		_count += 4;
		
		return ret;
	}

	@Override
	public long readLong() 
		throws IOException 
	{
		long ret = baToLong(_buff, _count);
		_count += 8;
		
		return ret;
	}

	@Override
	public float readFloat() throws IOException {
		throw new IOException("Not supported.");
	}

	@Override
	public double readDouble() 
		throws IOException 
	{
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
		throw new IOException("Not supported.");
	}
	
    ///////////////////////////////////////////////
    // Implementation of MatrixBlockDSMDataOutput
    ///////////////////////////////////////////////	
	
	@Override
	public int readDoubleArray(int len, double[] varr) 
		throws IOException 
	{
		//counter for non-zero elements
		int nnz = 0;
		
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
				for( int j=0; j<lnnz; j++ ) 
				{	
					int aix = baToInt(_buff, _count);
					long tmp = baToLong(_buff, _count+4);
					double aval = Double.longBitsToDouble( tmp );
					arow.append(aix, aval);
					_count+=12;
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
