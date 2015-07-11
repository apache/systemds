/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.data;

import java.io.Serializable;

import org.apache.spark.broadcast.Broadcast;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;

public class PartitionedBroadcast implements Serializable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -6941992543059285034L;

	//private Broadcast<MatrixBlock> _bc = null;
	private MatrixBlock[] _partBlocks = null; 
	private int _nrblks = -1;
	private int _ncblks = -1;
	
	public PartitionedBroadcast(Broadcast<MatrixBlock> bc, int brlen, int bclen) 
	{
		//get the input matrix block
		//_bc = bc;
		MatrixBlock mb = bc.value();
		int rlen = mb.getNumRows();
		int clen = mb.getNumColumns();
		
		//partitioning input broadcast
		try
		{
			_nrblks = (int)Math.ceil((double)rlen/brlen);
			_ncblks = (int)Math.ceil((double)clen/bclen);
			_partBlocks = new MatrixBlock[_nrblks * _ncblks];
			
			for( int i=0, ix=0; i<_nrblks; i++ )
				for( int j=0; j<_ncblks; j++, ix++ )
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
	
	public int getNumRowBlocks() {
		return _nrblks;
	}
	
	public int getNumColumnBlocks() {
		return _ncblks;
	}
	
	public MatrixBlock getMatrixBlock(int rowIndex, int colIndex) 
	{
		return _partBlocks[(rowIndex-1)*_ncblks + (colIndex-1)];
	}
}
