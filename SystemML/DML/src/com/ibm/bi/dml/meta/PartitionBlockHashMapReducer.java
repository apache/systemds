/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class PartitionBlockHashMapReducer extends MapReduceBase 
implements Reducer<BlockHashMapMapOutputKey, BlockHashMapMapOutputValue, MatrixIndexes, MatrixValue>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected MultipleOutputs multipleOutputs;
	PartitionParams pp = new PartitionParams() ;
	
	@Override
	public void reduce(BlockHashMapMapOutputKey key, Iterator<BlockHashMapMapOutputValue> values,
			OutputCollector<MatrixIndexes, MatrixValue> out, Reporter reporter)
	throws IOException {
		//reconstruct the matrix block from the subrows/subcols
		MatrixIndexes indexes = new MatrixIndexes(key.blky + 1, key.blkx + 1);		//systemml matrixblks start from (1,1)
		MatrixBlock thisblock = new MatrixBlock(pp.rows_in_block, pp.columns_in_block, true); //presume sparse
		
		//this is the even-more-eff cell kv pairs version! 1/c vs sparsity decision.
		while(values.hasNext()) {
			BlockHashMapMapOutputValue tmp = new BlockHashMapMapOutputValue(values.next());
			if(pp.isColumn == false)
				thisblock.setValue(tmp.auxdata, tmp.locator, tmp.cellvalue); 
			else
				thisblock.setValue(tmp.locator, tmp.auxdata, tmp.cellvalue);
		}
		
		/*
		//TODO: recheck about maxrow vs rlen!
		while(values.hasNext()) {
			BlockHashMapMapOutputValue tmp = new BlockHashMapMapOutputValue(values.next());
			MatrixBlock subblock = tmp.blk;
			int subrowcolid = tmp.auxdata;	//in this case, longdata is the subrowid/subcolid
			//the foll method is a more efficient method of transfering data across presumed sparse matrices
			boolean issparse = subblock.isInSparseFormat();
			if((issparse == true) && (subblock.getSparseMap() != null)) {
				Iterator<Entry<CellIndex, Double>> iter = subblock.getSparseMap().entrySet().iterator();
				while(iter.hasNext()) {
					Entry<CellIndex, Double> e = iter.next();
					if(e.getValue() == 0)
						continue;
					if(pp.isColumn == false)
						thisblock.setValue(subrowcolid, e.getKey().column, e.getValue()); //TODO: cellindex has row/col as ints!! can overflow!
					else
						thisblock.setValue(e.getKey().row, subrowcolid, e.getValue());
				}
			}
			else {
				double[] darray = subblock.getDenseArray();
				long limit = subblock.getNumRows() * subblock.getNumColumns();	//one of the rlen/clen will be 1
				if (darray != null) {
					for(int dd=0; dd<limit; dd++) {
						if(darray[dd] ==0)
							continue;
						if(pp.isColumn == false)
							thisblock.setValue(subrowcolid, dd, darray[dd]); //TODO: array index is only ints!! can overflow!!
						else
							thisblock.setValue(dd, subrowcolid, darray[dd]);
					}
				}
			}
			
			
			//this prev method of copying could be inefficient, since the matrix is presumed sparse
			//int nk = (pp.isColumn == false) ? subblock.getNumColumns() : subblock.getNumRows();
			//for(int c=0; c<nk; c++) {
			//	if(pp.isColumn == false)
			//		thisblock.setValue(subrowcolid, c, subblock.getValue(0, c));
			//	else
			//		thisblock.setValue(c, subrowcolid, subblock.getValue(c, 0));
			//}
			
			reporter.incrCounter("counter", "" + key.foldid, 1) ;	//incrmt num subrow/colblks counted for use in driver!
			if(pp.isColumn == false)	//subrowblk
				thisblock.setMaxColumn(subblock.getMaxColumn()); //check maxrows/maxcols of blks accordingly
			else	//subcolblk
				thisblock.setMaxRow(subblock.getMaxRow());
		}*/
		
		try {
			thisblock.examSparsity();
		} catch (DMLRuntimeException e) {
			// TODO Auto-generated catch block
			throw new IOException(e);
		}	//refactor based on actual sparsity
		//System.out.println("$$$$$$$ Blkhashmpreducer adding to fold " + key.foldid + " indexes: " + indexes.toString());
		reporter.incrCounter("counter", "" + key.foldid, 1) ;	//we need to use this in driver!
		multipleOutputs.getCollector("" + key.foldid, reporter).collect(indexes, thisblock) ;
	}

	public void close() throws IOException {
		multipleOutputs.close();
	}
	@Override
	public void configure(JobConf job) {
		multipleOutputs = new MultipleOutputs(job);
		pp = MRJobConfiguration.getPartitionParams(job);
	}
}