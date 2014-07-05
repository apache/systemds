/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.caching;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.FileNotFoundException;
import java.io.IOException;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;

/**
 * Wrapper for WriteBuffer byte array per matrix in order to
 * support matrix serialization outside global lock.
 * 
 */
public class ByteBuffer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	private boolean _serialized;	
	private boolean _sparse;
	private long _size;
	
	protected byte[]       _bdata = null; //sparse matrix
	protected MatrixBlock  _mdata = null; //dense matrix
	
	public ByteBuffer( long size )
	{
		_size = size;
		_serialized = false;
	}
	
	/**
	 * 
	 * @param mb
	 * @throws IOException
	 */
	public void serializeMatrix( MatrixBlock mb ) 
		throws IOException
	{	
		boolean sparseSrc = mb.isInSparseFormat(); //current representation
		boolean sparseTrgt = mb.evalSparseFormatOnDisk(); //intended target representation
		_sparse = sparseTrgt;
		
		try
		{
			if( _sparse ) //SPARSE/DENSE -> SPARSE
			{
				//deep serialize (for compression)
				if( CacheableData.CACHING_BUFFER_PAGECACHE )
					_bdata = PageCache.getPage((int)_size);
				if( _bdata==null )
					_bdata = new byte[(int)_size];
				DataOutput dout = new CacheDataOutput(_bdata);
				mb.write(dout);
			}
			else //SPARSE/DENSE -> DENSE
			{
				//change representation (if required), incl. free sparse
				//(in-memory representation, if dense on disk than if will
				//be guaranteed to be dense in memory as well)
				if( sparseSrc ) 
					mb.examSparsity(); 
				
				//shallow serialize
				_mdata = mb;
			}
		}
		catch(Exception ex)
		{
			throw new IOException("Failed to serialize matrix block.", ex);
		}
		
		_serialized = true;
	}
	
	/**
	 * 
	 * @return
	 * @throws IOException
	 */
	public MatrixBlock deserializeMatrix() 
		throws IOException
	{
		MatrixBlock ret = null;
		
		if( _sparse )
		{
			ByteArrayInputStream bis = new ByteArrayInputStream(_bdata);
			DataInputStream din = new DataInputStream(bis); 
			ret = new MatrixBlock();
			ret.readFields(din);
		}
		else
		{
			ret = _mdata;
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param fname
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void evictBuffer( String fname ) 
		throws FileNotFoundException, IOException
	{
		if( _sparse )
		{
			//write out byte serialized array
			LocalFileUtils.writeByteArrayToLocal(fname, _bdata);
		}
		else
		{
			//serialize matrix to output stream
			LocalFileUtils.writeMatrixBlockToLocal(fname, _mdata);
		}
	}
	
	/**
	 * Returns the buffer size in bytes.
	 * 
	 * @return
	 */
	public long getSize()
	{
		return _size;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isInSparseFormat()
	{
		return _sparse;
	}
	
	public void freeMemory()
	{
		//clear strong references to buffer/matrix
		if( _sparse )
		{
			if( CacheableData.CACHING_BUFFER_PAGECACHE )
				PageCache.putPage(_bdata);
			_bdata = null;
		}
		else
		{
			_mdata = null;
		}
	}
	
	/**
	 * 
	 */
	public void checkSerialized()
	{
		if( _serialized )
			return;
		
		while( !_serialized )
		{
			try{Thread.sleep(1);} catch(Exception e) {}
		}
	}
	
	/**
	 * Determines if byte buffer can hold the given size given this specific matrix block.
	 * This call is consistent with 'serializeMatrix' and allows for internal optimization
	 * according to dense/sparse representation.
	 * 
	 * @param size
	 * @param mb
	 * @return
	 */
	public static boolean isValidCapacity( long size, MatrixBlock mb )
	{
		boolean sparseTrgt = mb.evalSparseFormatOnDisk(); //intended target representation
		
		if( sparseTrgt ) //SPARSE
		{
			// since sparse matrix blocks are serialized into a byte representation
			// the buffer buffer can hold at most 2GB in size 
			return ( size <= Integer.MAX_VALUE );
		}
		else //DENSE
		{
			// since for dense matrix blocks we use a shallow serialize (strong reference), 
			// the byte buffer can hold any size (currently upper bounded by 16GB) 
			return true;
		}
	}
}
