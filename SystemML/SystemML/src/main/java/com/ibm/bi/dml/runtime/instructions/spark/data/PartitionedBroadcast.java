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

	//internal configuration parameters
	private static final boolean LAZY_BROADCAST_PARTITIONING = true;
	
	private Broadcast<MatrixBlock> _bc = null;
	private MatrixBlock[] _partBlocks = null; 
	private int _nrblks = -1;
	private int _ncblks = -1;
	private int _brlen = -1;
	private int _bclen = -1;
	
	public PartitionedBroadcast(Broadcast<MatrixBlock> bc, int brlen, int bclen) 
	{
		//get the input matrix block
		MatrixBlock mb = bc.value();
		int rlen = mb.getNumRows();
		int clen = mb.getNumColumns();
		
		//partitioning input broadcast
		_nrblks = (int)Math.ceil((double)rlen/brlen);
		_ncblks = (int)Math.ceil((double)clen/bclen);
		_partBlocks = new MatrixBlock[_nrblks * _ncblks];
			
		if( !LAZY_BROADCAST_PARTITIONING )
		{
			try
			{
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
		else
		{
			//keep required information for lazy partitioning
			_bc = bc;
			_brlen = brlen;
			_bclen = bclen;
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
		int rix = rowIndex - 1;
		int cix = colIndex - 1;
		
		if( LAZY_BROADCAST_PARTITIONING ) {
			//create matrix block partition on demand
			if( _partBlocks[rix*_ncblks + cix] == null )
			{
				try
				{
					MatrixBlock mb = _bc.value();
					MatrixBlock tmp = new MatrixBlock();
					mb.sliceOperations(rix*_brlen+1, Math.min((rix+1)*_brlen, mb.getNumRows()), 
						           cix*_bclen+1, Math.min((cix+1)*_bclen, mb.getNumColumns()), tmp);	
					_partBlocks[rix*_ncblks + cix] = tmp;
				}
				catch(Exception ex) {
					throw new RuntimeException("Failed partitioning of broadcast variable input.", ex);
				}
			}
		}
		
		return _partBlocks[rix*_ncblks + cix];
	}
}
