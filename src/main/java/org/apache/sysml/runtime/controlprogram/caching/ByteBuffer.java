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
 * Wrapper for WriteBuffer byte array per matrix/frame in order to
 * support matrix/frame serialization outside global lock.
 * 
 */
public class ByteBuffer
{
	private boolean _serialized;	
	private boolean _shallow;
	private long _size;
	
	protected byte[]     _bdata = null; //sparse matrix
	protected CacheBlock _cdata = null; //dense matrix/frame
	
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
	public void serializeBlock( CacheBlock cb ) 
		throws IOException
	{	
		_shallow = cb.isShallowSerialize();
		
		try
		{
			if( !_shallow ) //SPARSE/DENSE -> SPARSE
			{
				//deep serialize (for compression)
				if( CacheableData.CACHING_BUFFER_PAGECACHE )
					_bdata = PageCache.getPage((int)_size);
				if( _bdata==null )
					_bdata = new byte[(int)_size];
				DataOutput dout = new CacheDataOutput(_bdata);
				cb.write(dout);
			}
			else //SPARSE/DENSE -> DENSE
			{
				//special handling sparse matrix blocks whose serialized representation
				//is dense; change representation (if required), incl. free sparse
				if( cb instanceof MatrixBlock && ((MatrixBlock)cb).isInSparseFormat() )
					((MatrixBlock)cb).examSparsity();
				
				//shallow serialize
				_cdata = cb;
			}
		}
		catch(Exception ex)
		{
			throw new IOException("Failed to serialize cache block.", ex);
		}
		
		_serialized = true;
	}
	
	/**
	 * 
	 * @return
	 * @throws IOException
	 */
	public CacheBlock deserializeBlock() 
		throws IOException
	{
		CacheBlock ret = null;
		
		if( !_shallow ) { //sparse matrix 
			CacheDataInput din = new CacheDataInput(_bdata);
			ret = new MatrixBlock();
			ret.readFields(din);
		}
		else { //dense matrix/frame
			ret = _cdata;
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
		if( !_shallow ) {
			//write out byte serialized array
			LocalFileUtils.writeByteArrayToLocal(fname, _bdata);
		}
		else {
			//serialize cache block to output stream
			LocalFileUtils.writeCacheBlockToLocal(fname, _cdata);
		}
	}
	
	/**
	 * Returns the buffer size in bytes.
	 * 
	 * @return
	 */
	public long getSize() {
		return _size;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isShallow() {
		return _shallow;
	}
	
	public void freeMemory()
	{
		//clear strong references to buffer/matrix
		if( !_shallow ) {
			if( CacheableData.CACHING_BUFFER_PAGECACHE )
				PageCache.putPage(_bdata);
			_bdata = null;
		}
		else {
			_cdata = null;
		}
	}
	
	/**
	 * 
	 */
	public void checkSerialized()
	{
		//check if already serialized
		if( _serialized )
			return;
		
		//robust checking until serialized
		while( !_serialized ) {
			try{Thread.sleep(1);} catch(Exception e) {}
		}
	}
	
	/**
	 * Determines if byte buffer can hold the given size given this specific cache block.
	 * This call is consistent with 'serializeBlock' and allows for internal optimization
	 * according to dense/sparse representation.
	 * 
	 * @param size
	 * @param mb
	 * @return
	 */
	public static boolean isValidCapacity( long size, CacheBlock cb )
	{
		if( !cb.isShallowSerialize() ) { //SPARSE matrix blocks
			// since cache blocks are serialized into a byte representation
			// the buffer buffer can hold at most 2GB in size 
			return ( size <= Integer.MAX_VALUE );	
		}
		else {//DENSE matrix / frame blocks
			// since for dense matrix blocks we use a shallow serialize (strong reference), 
			// the byte buffer can hold any size (currently upper bounded by 16GB) 
			return true;
		}
	}
}
