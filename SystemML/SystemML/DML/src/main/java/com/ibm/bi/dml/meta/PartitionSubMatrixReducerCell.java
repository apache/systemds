/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
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

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.data.PartialBlock;
import com.ibm.bi.dml.runtime.matrix.data.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class PartitionSubMatrixReducerCell extends MapReduceBase 
implements Reducer<TaggedFirstSecondIndexes, PartialBlock, MatrixIndexes, MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//private static final Log LOG = LogFactory.getLog(PartitionSubMatrixReducerCell.class);
	protected MultipleOutputs multipleOutputs;
	
	private MatrixBlock blockBuffer = new MatrixBlock() ;
	MatrixIndexes indexes = new MatrixIndexes() ;
	int brlen, bclen; long rlen, clen ;
	long[] rowLengths, colLengths ;
	
	@Override
	public void reduce(TaggedFirstSecondIndexes pair, Iterator<PartialBlock> values,
			OutputCollector<MatrixIndexes, MatrixBlock> out, Reporter reporter)
			throws IOException {
		
		indexes.setIndexes(pair.getFirstIndex(), pair.getSecondIndex()) ;
		int realBrlen=(int)Math.min((long)brlen, rowLengths[pair.getTag()]-(indexes.getRowIndex()-1)*brlen);
		int realBclen=(int)Math.min((long)bclen, colLengths[pair.getTag()]-(indexes.getColumnIndex()-1)*bclen);
		
		blockBuffer.reset(realBrlen, realBclen);
		while(values.hasNext())
		{
			PartialBlock partial=values.next();
			blockBuffer.setValue(partial.getRowIndex(), partial.getColumnIndex(), partial.getValue());
		}
		multipleOutputs.getCollector(""+pair.getTag(), reporter).collect(indexes, blockBuffer) ;
	}

	public void close() throws IOException {
		multipleOutputs.close();
	}
	
	@Override
	public void configure(JobConf job) {
		//get input converter information
		brlen=MRJobConfiguration.getNumRowsPerBlock(job, (byte)0);
		bclen=MRJobConfiguration.getNumColumnsPerBlock(job, (byte)0);
		rlen = MRJobConfiguration.getNumRows(job, (byte) 0) ;
		clen = MRJobConfiguration.getNumColumns(job, (byte) 0) ;
		
		multipleOutputs = new MultipleOutputs(job) ;
		PartitionParams pp = MRJobConfiguration.getPartitionParams(job) ;
		Pair<long[],long[]> pair = pp.getRowAndColumnLengths(rlen, clen) ;
		rowLengths = pair.getKey() ;
		colLengths = pair.getValue() ;
	}
}
