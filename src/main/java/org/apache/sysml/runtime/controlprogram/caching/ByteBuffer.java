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

import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.LocalFileUtils;

/**
 * Wrapper for WriteBuffer byte array per matrix in order to
 * support matrix serialization outside global lock.
 * 
 */
public class ByteBuffer
{
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
			//ByteArrayInputStream bis = new ByteArrayInputStream(_bdata);
			//DataInputStream din = new DataInputStream(bis); 
			CacheDataInput din = new CacheDataInput(_bdata);
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
	 * @throws IOException
	 */
	public void evictBuffer( String fname ) 
		throws IOException
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
