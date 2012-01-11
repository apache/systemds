package dml.meta;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import dml.parser.DataIdentifier;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.io.PartialBlock;
import dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import dml.runtime.matrix.mapred.MRJobConfiguration;

public class PartitionSubMatrixReducerCell extends MapReduceBase 
implements Reducer<TaggedFirstSecondIndexes, PartialBlock, MatrixIndexes, MatrixBlock> {

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
