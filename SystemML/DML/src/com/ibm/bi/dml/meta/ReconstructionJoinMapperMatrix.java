package com.ibm.bi.dml.meta;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.io.Converter;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class ReconstructionJoinMapperMatrix extends MapReduceBase
implements Mapper<Writable, Writable, LongWritable, ReconstructionJoinMapOutputValue> {
	
	private Converter inputConverter=null;
	PartitionParams pp = new PartitionParams() ;
	int brlen, bclen ;
	MultipleOutputs multipleOutputs ;

	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<LongWritable, ReconstructionJoinMapOutputValue> out, Reporter reporter)
	throws IOException {
		inputConverter.setBlockSize(brlen, bclen); // 2 x 2 matrix blocks..
		inputConverter.convert(rawKey, rawValue);
		while(inputConverter.hasNext()) {
			Pair<MatrixIndexes, MatrixBlock> pair=inputConverter.next();
			//bmm.execute(pair, reporter, out) ;
			long blky = pair.getKey().getRowIndex() - 1;		//systemml matrixblks start from (1,1)
			MatrixBlock matvalue = pair.getValue();
			int nrows = matvalue.getNumRows();	//assuming W is a col matrix
			int rpb = pp.rows_in_block;			//TODO: assuming this is the general block y dimension
			for(int r=0; r<nrows; r++) {
				LongWritable outkey = new LongWritable(blky * rpb + r);	//absol rowid
				ReconstructionJoinMapOutputValue outval = new ReconstructionJoinMapOutputValue();
				outval.rowid = -1;	//this means it is matrix elem kv pair
				outval.entry = matvalue.getValue(r, 0);	//single col matrix
				out.collect(outkey, outval);	//one output key-val pair for each absol rowid
			}
		}
	}
	
	public void close() throws IOException  {
		multipleOutputs.close();
	}
	
	@Override
	public void configure(JobConf job) {
		multipleOutputs = new MultipleOutputs(job) ;
		//get input converter information
		inputConverter=MRJobConfiguration.getInputConverter(job, (byte)0);
		brlen=MRJobConfiguration.getNumRowsPerBlock(job, (byte)0);
		bclen=MRJobConfiguration.getNumColumnsPerBlock(job, (byte)0);		
		pp = MRJobConfiguration.getPartitionParams(job) ;
	}
}
