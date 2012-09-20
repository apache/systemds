package com.ibm.bi.dml.meta;
//<Arun>
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.utils.DMLRuntimeException;


public abstract class BlockJoinMapperMethodIDTable {
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