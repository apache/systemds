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

package org.apache.sysml.runtime.instructions.spark.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.util.FastBufferedDataInputStream;
import org.apache.sysml.runtime.util.FastBufferedDataOutputStream;
import org.apache.sysml.runtime.util.IndexRange;

/**
 * The main purpose of this class is to provide a handle for partitioned frame blocks, to be used
 * as broadcasts. Distributed tasks require block-partitioned broadcasts but a lazy partitioning per
 * task would create instance-local copies and hence replicate broadcast variables which are shared
 * by all tasks within an executor.  
 * 
 */
public class PartitionedFrameBlock extends PartitionedBlock implements Externalizable
{

	public PartitionedFrameBlock() {
		//do nothing (required for Externalizable)
	}
	
	public PartitionedFrameBlock(FrameBlock mb, int brlen, int bclen) 
	{
		//get the input frame block
		int rlen = mb.getNumRows();
		int clen = mb.getNumColumns();
		
		//partitioning input broadcast
		_rlen = rlen;
		_clen = clen;
		_brlen = brlen;
		_bclen = bclen;
		
		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		_partBlocks = new FrameBlock[nrblks * ncblks];
		
		try
		{
			for( int i=0, ix=0; i<nrblks; i++ )
				for( int j=0; j<ncblks; j++, ix++ )
				{
					FrameBlock tmp = new FrameBlock();
					mb.sliceOperations(i*_brlen, Math.min((i+1)*_brlen, rlen)-1, 
							           j*_bclen, Math.min((j+1)*_bclen, clen)-1, tmp);
					_partBlocks[ix] = tmp;
				}
		}
		catch(Exception ex) {
			throw new RuntimeException("Failed partitioning of broadcast variable input.", ex);
		}
		
		_offset = 0;
	}
	
	/**
	 * 
	 * @param rowIndex
	 * @param colIndex
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public FrameBlock getFrameBlock(int rowIndex, int colIndex) 
		throws DMLRuntimeException 
	{
		//check for valid block index
		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		if( rowIndex <= 0 || rowIndex > nrblks || colIndex <= 0 || colIndex > ncblks ) {
			throw new DMLRuntimeException("Block indexes ["+rowIndex+","+colIndex+"] out of range ["+nrblks+","+ncblks+"]");
		}
		
		//get the requested frame block
		int rix = rowIndex - 1;
		int cix = colIndex - 1;
		int ix = rix*ncblks+cix - _offset;
		return (FrameBlock) _partBlocks[ ix ];
	}
	
	/**
	 * 
	 * @param offset
	 * @param numBlks
	 * @return
	 */
	public PartitionedFrameBlock createPartition( int offset, int numBlks )
	{
		PartitionedFrameBlock ret = new PartitionedFrameBlock();
		ret._rlen = _rlen;
		ret._clen = _clen;
		ret._brlen = _brlen;
		ret._bclen = _bclen;
		ret._partBlocks = new FrameBlock[numBlks];
		ret._offset = offset;
		
		System.arraycopy(_partBlocks, offset, ret._partBlocks, 0, numBlks);
		
		return ret;
	}

	/**
	 * Utility for slice operations over partitioned matrices, where the index range can cover
	 * multiple blocks. The result is always a single result frame block. All semantics are 
	 * equivalent to the core frame block slice operations. 
	 * 
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @param frameBlock
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public FrameBlock sliceOperations(IndexRange ixrangeGbl, long rl, long ru, long cl, long cu, FrameBlock frameBlock) 
		throws DMLRuntimeException 
	{
		int lrl = (int) rl;
		int lru = (int) ru;
		int lcl = (int) cl;
		int lcu = (int) cu;
		
		ArrayList<Pair<Long, FrameBlock>> allBlks = new ArrayList<Pair<Long,FrameBlock>>();
		int start_iix = (lrl-1)/_brlen+1;
		int end_iix = (lru-1)/_brlen+1;
		int start_jix = (lcl-1)/_bclen+1;
		int end_jix = (lcu-1)/_bclen+1;
				
		FrameBlock in = null;
		for( int iix = start_iix; iix <= end_iix; iix++ )
			for(int jix = start_jix; jix <= end_jix; jix++)		
			{
				in = getFrameBlock(iix, jix);
				Pair<Long, FrameBlock> lfp = new Pair<Long, FrameBlock>(new Long(((iix-1)*_brlen)+1), in);
				ArrayList<Pair<Long, FrameBlock>> outlist = new ArrayList<Pair<Long, FrameBlock>>();
				IndexRange ixrange = new IndexRange(rl, ru, cl, cu);
				OperationsOnMatrixValues.performSlice(lfp, ixrange, _brlen, _bclen, outlist);
				allBlks.addAll(outlist);
			}
		
		if(allBlks.size() == 1) {
			return allBlks.get(0).getValue();
		}
		else {
			//allocate output frame
//			FrameBlock ret = new FrameBlock(lcu-lcl+1, ValueType.STRING);
			FrameBlock ret = new FrameBlock(in.getSchema());
			ret.ensureAllocatedColumns((int) (ru-rl+1));
			
			for(Pair<Long, FrameBlock> kv : allBlks) {
				ret.merge(kv.getValue(), (int) (kv.getKey()-rl), true);
			}
			return ret;
		} 
	}
	
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient broadcast deserialization. 
	 * 
	 * @param is
	 * @throws IOException
	 */
	public void readExternal(ObjectInput is) 
		throws IOException
	{
		DataInput dis = is;
		
		if( is instanceof ObjectInputStream ) {
			//fast deserialize of dense/sparse blocks
			ObjectInputStream ois = (ObjectInputStream)is;
			dis = new FastBufferedDataInputStream(ois);
		}
		
		readHeaderAndPayload(dis);
	}
	
	/**
	 * Redirects the default java serialization via externalizable to our default 
	 * hadoop writable serialization for efficient broadcast serialization. 
	 * 
	 * @param is
	 * @throws IOException
	 */
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
	
	/**
	 * 
	 * @param dos
	 * @throws IOException 
	 */
	private void writeHeaderAndPayload(DataOutput dos) 
		throws IOException
	{
		dos.writeLong(_rlen);
		dos.writeLong(_clen);
		dos.writeInt(_brlen);
		dos.writeInt(_bclen);
		dos.writeInt(_offset);
		dos.writeInt(_partBlocks.length);
		for( FrameBlock fb : (FrameBlock[]) _partBlocks )
			fb.write(dos);
	}

	/**
	 * 
	 * @param din
	 * @throws IOException 
	 */
	private void readHeaderAndPayload(DataInput dis) 
		throws IOException
	{
		_rlen = dis.readInt();
		_clen = dis.readInt();
		_brlen = dis.readInt();
		_bclen = dis.readInt();
		_offset = dis.readInt();
		
		int len = dis.readInt();
		_partBlocks = new FrameBlock[len];
		
		for( int i=0; i<len; i++ ) {
			_partBlocks[i] = new FrameBlock();
			_partBlocks[i].readFields(dis);
		}
	}
}
