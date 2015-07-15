/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.FastBufferedDataInputStream;
import com.ibm.bi.dml.runtime.util.FastBufferedDataOutputStream;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

/**
 * The main purpose of this class is to provide a handle for partitioned matrix blocks, to be used
 * as broadcasts. Distributed tasks require block-partitioned broadcasts but a lazy partitioning per
 * task would create instance-local copies and hence replicate broadcast variables which are shared
 * by all tasks within an executor.  
 * 
 */
public class PartitionedMatrixBlock implements Externalizable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

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
	 */
	public MatrixBlock getMatrixBlock(int rowIndex, int colIndex) 
	{
		int rix = rowIndex - 1;
		int cix = colIndex - 1;
		int ncblks = getNumColumnBlocks();
		
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
		
		//allocate output matrix
		MatrixBlock ret = new MatrixBlock(lru-lrl+1, lcu-lcl+1, false);
		
		//slice operations (at most 4 blocks)
		for( int i = lrl; i <= lru; i+=_brlen )
			for(int j = lcl; j <= lcu; j+=_bclen)
			{
				//get the current block
				int iix = (i-1)/_brlen+1;
				int jix = (j-1)/_bclen+1;
				MatrixBlock in = getMatrixBlock(iix, jix);
				
				//slice out relevant portion of current block
				int ix1 = UtilFunctions.cellInBlockCalculation(i, _brlen)+1;
				int ix2 = UtilFunctions.cellInBlockCalculation(Math.min(i+_brlen, lru), _brlen)+1;
				int ix3 = UtilFunctions.cellInBlockCalculation(j, _bclen)+1;
				int ix4 = UtilFunctions.cellInBlockCalculation(Math.min(j+_bclen, lcu), _bclen)+1;
				MatrixBlock in2 = in.sliceOperations(ix1, ix2, ix3, ix4, new MatrixBlock());
				
				//left indexing temporary block into result
				ret.leftIndexingOperations(in2, i-lrl+1, Math.min(i-lrl+1+_brlen, lru-lrl+1), 
						                   j-lcl+1, Math.min(j-lcl+1+_bclen, lcu-lcl+1), ret, true);
			}
			
		return ret;
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
