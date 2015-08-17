/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark.data;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.spark.utils.SparkUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.util.FastBufferedDataInputStream;
import com.ibm.bi.dml.runtime.util.FastBufferedDataOutputStream;
import com.ibm.bi.dml.runtime.util.IndexRange;

/**
 * The main purpose of this class is to provide a handle for partitioned matrix blocks, to be used
 * as broadcasts. Distributed tasks require block-partitioned broadcasts but a lazy partitioning per
 * task would create instance-local copies and hence replicate broadcast variables which are shared
 * by all tasks within an executor.  
 * 
 */
public class PartitionedMatrixBlock implements Externalizable
{

	private static final long serialVersionUID = -5706923809800365593L;

	private MatrixBlock[] _partBlocks = null; 
	private int _rlen = -1;
	private int _clen = -1;
	private int _brlen = -1;
	private int _bclen = -1;
	
	public PartitionedMatrixBlock() {
		//do nothing (required for Externalizable)
	}
	
	public PartitionedMatrixBlock(MatrixBlock mb, int brlen, int bclen) 
	{
		//get the input matrix block
		int rlen = mb.getNumRows();
		int clen = mb.getNumColumns();
		
		//partitioning input broadcast
		_rlen = rlen;
		_clen = clen;
		_brlen = brlen;
		_bclen = bclen;
		
		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		_partBlocks = new MatrixBlock[nrblks * ncblks];
		
		try
		{
			for( int i=0, ix=0; i<nrblks; i++ )
				for( int j=0; j<ncblks; j++, ix++ )
				{
					MatrixBlock tmp = new MatrixBlock();
					mb.sliceOperations(i*brlen+1, Math.min((i+1)*brlen, rlen), 
							           j*bclen+1, Math.min((j+1)*bclen, clen), tmp);
					_partBlocks[ix] = tmp;
				}
		}
		catch(Exception ex) {
			throw new RuntimeException("Failed partitioning of broadcast variable input.", ex);
		}		
	}
	
	/**
	 * 
	 * @return
	 */
	public int getNumRowBlocks() 
	{
		return (int)Math.ceil((double)_rlen/_brlen);
	}
	
	public long getNumRows() {
		return _rlen;
	}
	
	public long getNumCols() {
		return _clen;
	}
	
	/**
	 * 
	 * @return
	 */
	public int getNumColumnBlocks() 
	{
		return (int)Math.ceil((double)_clen/_bclen);
	}
	
	/**
	 * 
	 * @param rowIndex
	 * @param colIndex
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public MatrixBlock getMatrixBlock(int rowIndex, int colIndex) 
		throws DMLRuntimeException 
	{
		//check for valid block index
		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		if( rowIndex <= 0 || rowIndex > nrblks || colIndex <= 0 || colIndex > ncblks ) {
			throw new DMLRuntimeException("Block indexes ["+rowIndex+","+colIndex+"] out of range ["+nrblks+","+ncblks+"]");
		}
		
		//get the requested matrix block
		int rix = rowIndex - 1;
		int cix = colIndex - 1;
		return _partBlocks[rix*ncblks + cix];
	}
	
	/**
	 * 
	 * @return
	 */
	public long estimateSizeInMemory()
	{
		long ret = 8; //header
		ret += 32;    //block array
		
		if( _partBlocks != null )
			for( MatrixBlock mb : _partBlocks )
				ret += mb.estimateSizeInMemory();
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public long estimateSizeOnDisk()
	{
		long ret = 8; //header
		
		if( _partBlocks != null )
			for( MatrixBlock mb : _partBlocks )
				ret += mb.estimateSizeOnDisk();
		
		return ret;
	}
	

	/**
	 * Utility for slice operations over partitioned matrices, where the index range can cover
	 * multiple blocks. The result is always a single result matrix block. All semantics are 
	 * equivalent to the core matrix block slice operations. 
	 * 
	 * @param rl
	 * @param ru
	 * @param cl
	 * @param cu
	 * @param matrixBlock
	 * @return
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	public MatrixBlock sliceOperations(long rl, long ru, long cl, long cu, MatrixBlock matrixBlock) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		int lrl = (int) rl;
		int lru = (int) ru;
		int lcl = (int) cl;
		int lcu = (int) cu;
		
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> allBlks = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		int start_iix = (lrl-1)/_brlen+1;
		int end_iix = (lru-1)/_brlen+1;
		int start_jix = (lcl-1)/_bclen+1;
		int end_jix = (lcu-1)/_bclen+1;
				
		for( int iix = start_iix; iix <= end_iix; iix++ )
			for(int jix = start_jix; jix <= end_jix; jix++)		
			{
				MatrixBlock in = getMatrixBlock(iix, jix);
				IndexedMatrixValue imv = new IndexedMatrixValue(new MatrixIndexes(iix, jix), in);
				
				ArrayList<IndexedMatrixValue> outlist = new ArrayList<IndexedMatrixValue>();
				IndexRange ixrange = new IndexRange(rl, ru, cl, cu);
				OperationsOnMatrixValues.performSlice(imv, ixrange, _brlen, _bclen, outlist);
				allBlks.addAll(SparkUtils.fromIndexedMatrixBlock(outlist));
			}
		
		if(allBlks.size() == 1) {
			return allBlks.get(0)._2;
		}
		else {
			//allocate output matrix
			MatrixBlock ret = new MatrixBlock(lru-lrl+1, lcu-lcl+1, false);
			for(Tuple2<MatrixIndexes, MatrixBlock> kv : allBlks) {
				ret.merge(kv._2, false);
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
		dos.writeInt(_rlen);
		dos.writeInt(_clen);
		dos.writeInt(_brlen);
		dos.writeInt(_bclen);
		for( MatrixBlock mb : _partBlocks )
			mb.write(dos);
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
		int nrblks = getNumRowBlocks();
		int ncblks = getNumColumnBlocks();
		_partBlocks = new MatrixBlock[nrblks * ncblks];
		
		for( int i=0; i<_partBlocks.length; i++ ){
			_partBlocks[i] = new MatrixBlock();
			_partBlocks[i].readFields(dis);
		}
	}
}
