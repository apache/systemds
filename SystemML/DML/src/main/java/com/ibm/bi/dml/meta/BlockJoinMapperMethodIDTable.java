/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;
//<Arun>
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;


public abstract class BlockJoinMapperMethodIDTable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	MatrixIndexes mi = new MatrixIndexes() ;
	PartitionParams pp ;
	MultipleOutputs multipleOutputs ;
	
	public BlockJoinMapperMethodIDTable () {
		mi = null;
		pp = null;
	}
	
	public BlockJoinMapperMethodIDTable(PartitionParams pp, MultipleOutputs multipleOutputs) {
		this.pp = pp ;
		this.multipleOutputs = multipleOutputs ;
	}
	
	abstract void execute(LongWritable key, WritableLongArray value, Reporter reporter, OutputCollector out) 
	throws IOException ;
		
	public MatrixBlock getSubRowBlock(MatrixBlock blk, int rownum) throws DMLRuntimeException {
		int ncols = blk.getNumColumns();
		MatrixBlock thissubrowblk = new MatrixBlock(1, ncols, true, blk.getNonZeros()/blk.getNumRows());	//presume sparse
		//populate subrowblock
		for(int c=0; c<ncols; c++) {
			thissubrowblk.setValue(rownum, c, blk.getValue(rownum, c));
		}
		thissubrowblk.examSparsity();	//refactor based on sparsity
		return thissubrowblk;
	}
}
//</Arun>