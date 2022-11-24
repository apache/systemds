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

package org.apache.sysds.runtime.controlprogram.caching;

import java.io.ByteArrayInputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.sysds.runtime.data.DenseBlockLDRB;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.LocalFileUtils;

/**
 * Wrapper for WriteBuffer byte array per matrix/frame in order to
 * support matrix/frame serialization outside global lock.
 * 
 */
public class ByteBuffer
{
	private volatile boolean _serialized;
	private volatile boolean _shallow;
	private volatile boolean _matrix;
	private final long _size;
	
	protected byte[]     _bdata = null; //sparse matrix
	protected CacheBlock<?> _cdata = null; //dense matrix/frame
	
	public ByteBuffer( long size ) {
		_size = size;
		_serialized = false;
	}

	public void serializeBlock( CacheBlock<?> cb ) 
		throws IOException
	{	
		_shallow = cb.isShallowSerialize(true);
		_matrix = (cb instanceof MatrixBlock);
		
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
				//convert to shallow serialize block if necessary
				if( !cb.isShallowSerialize() )
					cb.toShallowSerializeBlock();
				
				//shallow serialize
				_cdata = cb;
			}
		}
		catch(Exception ex) {
			throw new IOException("Failed to serialize cache block.", ex);
		}
		
		_serialized = true;
	}

	public CacheBlock<?> deserializeBlock() 
		throws IOException
	{
		CacheBlock<?> ret = null;
		
		if( !_shallow ) { //sparse matrix / string frame
			DataInput din = _matrix ? new CacheDataInput(_bdata) :
				new DataInputStream(new ByteArrayInputStream(_bdata));
			ret = _matrix ? new MatrixBlock() : new FrameBlock();
			ret.readFields(din);
		}
		else { //dense matrix/frame
			ret = _cdata;
		}
		
		return ret;
	}

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
	 * @return buffer size in bytes
	 */
	public long getSize() {
		return _size;
	}

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
	 * @param size the size
	 * @param cb cache block
	 * @return true if valid capacity
	 */
	public static boolean isValidCapacity( long size, CacheBlock<?> cb )
	{
		if( !cb.isShallowSerialize(true) ) { //SPARSE matrix blocks
			// since cache blocks are serialized into a byte representation
			// the buffer buffer can hold at most 2GB in size 
			return ( size <= DenseBlockLDRB.MAX_ALLOC );
		}
		else {//DENSE/SPARSE matrix / frame blocks
			// for dense and under special conditions also sparse matrix blocks 
			// we use a shallow serialize (strong reference), there is no additional
			// capacity constraint for serializing these blocks into byte arrays
			return true;
		}
	}
}
