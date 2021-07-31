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

package org.apache.sysds.runtime.instructions.spark.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.Arrays;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlockFactory;
import org.apache.sysds.runtime.util.FastBufferedDataInputStream;
import org.apache.sysds.runtime.util.FastBufferedDataOutputStream;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * This class is for partitioned matrix/frame blocks, to be used as broadcasts. 
 * Distributed tasks require block-partitioned broadcasts but a lazy partitioning 
 * per task would create instance-local copies and hence replicate broadcast 
 * variables which are shared by all tasks within an executor.  
 * 
 */
public class PartitionedBlock<T extends CacheBlock> implements Externalizable
{
	private static final long serialVersionUID = 1298817743064415129L;
	protected CacheBlock[] _partBlocks = null;
	protected long[] _dims = {-1, -1};
	protected int _blen = -1;
	protected int _offset = 0;
	
	public PartitionedBlock() {
		//do nothing (required for Externalizable)
	}
	
	@SuppressWarnings("unchecked")
	public PartitionedBlock(T block, int blen) 
	{
		//get the input frame block
		int rlen = block.getNumRows();
		int clen = block.getNumColumns();
		
		//partitioning input broadcast
		_dims = new long[]{rlen, clen};
		_blen = blen;
		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		int code = CacheBlockFactory.getCode(block);
		
		try {
			_partBlocks = new CacheBlock[nrblks * ncblks];
			Arrays.parallelSetAll(_partBlocks, index -> {
				int i = index / ncblks;
				int j = index % ncblks;
				T tmp = (T) CacheBlockFactory.newInstance(code);
				return block.slice(i * _blen, Math.min((i + 1) * _blen, rlen) - 1,
					j * _blen, Math.min((j + 1) * _blen, clen) - 1, false, tmp);
			});
		} catch(Exception ex) {
			throw new RuntimeException("Failed partitioning of broadcast variable input.", ex);
		}

		_offset = 0;
	}

	@SuppressWarnings("unchecked")
	public PartitionedBlock(T block, long[] dims, int blen)
	{
		//partitioning input broadcast
		_dims = dims;
		_blen = blen;
		int nblks = 1;
		for (int i = 0; i < dims.length; i++)
			nblks *= getNumDimBlocks(i);
		int code = CacheBlockFactory.getCode(block);

		try {
			_partBlocks = new CacheBlock[nblks];
			int div = nblks / getNumRowBlocks();
			Arrays.parallelSetAll(_partBlocks, index -> {
				int i = index / div;
				int j = index % div;
				T tmp = (T) CacheBlockFactory.newInstance(code);
				return block.slice(i * _blen, Math.min((i + 1) * _blen, (int)_dims[0]) - 1,
						j * _blen, Math.min((j + 1) * _blen, (int)_dims[1]) - 1, tmp);
			});
		} catch(Exception ex) {
			throw new RuntimeException("Failed partitioning of broadcast variable input.", ex);
		}

		_offset = 0;
	}

	public PartitionedBlock(int rlen, int clen, int blen) {
		//partitioning input broadcast
		_dims = new long[]{rlen, clen};
		_blen = blen;

		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		_partBlocks = new CacheBlock[nrblks * ncblks];
	}

	public PartitionedBlock<T> createPartition( int offset, int numBlks)
	{
		PartitionedBlock<T> ret = new PartitionedBlock<>();
		ret._dims = _dims.clone();
		ret._blen = _blen;
		ret._partBlocks = new CacheBlock[numBlks];
		ret._offset = offset;
		System.arraycopy(_partBlocks, offset, ret._partBlocks, 0, numBlks);
		
		return ret;
	}
	
	public long getNumRows() {
		return _dims[0];
	}
	
	public long getNumCols() {
		return _dims[1];
	}

	public long getDim(int i) {
		return _dims[i];
	}
	
	public long getBlocksize() {
		return _blen;
	}

	public int getNumRowBlocks() {
		return getNumDimBlocks(0);
	}

	public int getNumColumnBlocks() {
		return getNumDimBlocks(1);
	}

	public int getNumDimBlocks(int dim) {
		return (int)Math.ceil((double)_dims[dim]/_blen);
	}

	@SuppressWarnings("unchecked")
	public T getBlock(int rowIndex, int colIndex) {
		//check for valid block index
		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		if( rowIndex <= 0 || rowIndex > nrblks || colIndex <= 0 || colIndex > ncblks ) {
			throw new DMLRuntimeException("Block indexes ["+rowIndex+","+colIndex+"] out of range ["+nrblks+","+ncblks+"]");
		}
		
		//get the requested frame/matrix block
		int rix = rowIndex - 1;
		int cix = colIndex - 1;
		int ix = rix*ncblks+cix - _offset;
		return (T)_partBlocks[ix];
	}

	@SuppressWarnings("unchecked")
	public T getBlock(int[] ix) {
		long index = UtilFunctions.computeBlockNumber(ix, _dims, _blen);
		index -= _offset;
		return (T)_partBlocks[(int) index];
	}

	public void setBlock(int rowIndex, int colIndex, T block) {
		//check for valid block index
		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		if( rowIndex <= 0 || rowIndex > nrblks || colIndex <= 0 || colIndex > ncblks ) {
			throw new DMLRuntimeException("Block indexes ["+rowIndex+","+colIndex+"] out of range ["+nrblks+","+ncblks+"]");
		}
		
		//get the requested matrix block
		int rix = rowIndex - 1;
		int cix = colIndex - 1;
		int ix = rix*ncblks+cix - _offset;
		_partBlocks[ ix ] = block;	
	}

	public long getInMemorySize() {
		long ret = 24; //header
		ret += 32;    //block array
		
		if( _partBlocks != null )
			for( CacheBlock block : _partBlocks )
				ret += block.getInMemorySize();
		
		return ret;
	}

	public long getExactSerializedSize() {
		long ret = 24; //header
		
		if( _partBlocks != null )
			for( CacheBlock block : _partBlocks )
				ret += block.getExactSerializedSize();
		
		return ret;
	}
	
	public void clearBlocks() {
		_partBlocks = null;
	}

	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient broadcast deserialization. 
	 * 
	 * @param is object input
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void readExternal(ObjectInput is) 
		throws IOException
	{
		DataInput dis = is;
		
		int code = readHeader(dis);
		if( is instanceof ObjectInputStream && code == 0) {	// Apply only for MatrixBlock at this point as a temporary workaround
															// We will generalize this code by adding UTF functionality to support Frame
			//fast deserialize of dense/sparse blocks
			ObjectInputStream ois = (ObjectInputStream)is;
			dis = new FastBufferedDataInputStream(ois);
		}
		readPayload(dis, code);
	}
	
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient broadcast serialization. 
	 * 
	 * @param os object output
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void writeExternal(ObjectOutput os) 
		throws IOException
	{
		if( os instanceof ObjectOutputStream ) {
			//fast serialize of dense/sparse blocks
			ObjectOutputStream oos = (ObjectOutputStream)os;
			FastBufferedDataOutputStream fos = new FastBufferedDataOutputStream(oos);
			writeHeaderAndPayload(fos);
			fos.flush();
		}
		else {
			//default serialize (general case)
			writeHeaderAndPayload(os);	
		}
	}

	private void writeHeaderAndPayload(DataOutput dos) 
		throws IOException
	{
		dos.writeInt(_dims.length);
		for (long dim : _dims)
			dos.writeLong(dim);
		dos.writeInt(_blen);
		dos.writeInt(_offset);
		dos.writeInt(_partBlocks.length);
		dos.writeByte(CacheBlockFactory.getCode(_partBlocks[0]));
		
		for( CacheBlock block : _partBlocks )
			block.write(dos);
	}

	private int readHeader(DataInput dis) 
		throws IOException
	{
		int length = dis.readInt();
		_dims = new long[length];
		for (int i = 0; i < length; i++)
			_dims[i] = dis.readLong();
		_blen = dis.readInt();
		_offset = dis.readInt();
		int len = dis.readInt();
		int code = dis.readByte();
		
		_partBlocks = new CacheBlock[len];
		
		return code;
	}

	private void readPayload(DataInput dis, int code) 
		throws IOException
	{
		int len = _partBlocks.length;
		for( int i=0; i<len; i++ ) {
			_partBlocks[i] = CacheBlockFactory.newInstance(code);
			_partBlocks[i].readFields(dis);
		}
	}
}
