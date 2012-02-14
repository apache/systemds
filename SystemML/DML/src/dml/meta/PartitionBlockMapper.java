package dml.meta;

import java.io.IOException;

import org.apache.commons.math.random.Well1024a;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.util.MapReduceTool;

public class PartitionBlockMapper extends MapReduceBase
implements Mapper<Writable, Writable, IntWritable, MatrixBlock> {
	private Well1024a[] random ;
	private Well1024a currRandom ;
	
	private Converter inputConverter=null;
	PartitionParams pp = new PartitionParams() ;
	int brlen, bclen ;
	long numRows, numColumns ;
	BlockMapperMethod bmm ;
	MultipleOutputs multipleOutputs ;
	MatrixBlock block ;
	int domVal ;
	IntWritable obj = new IntWritable() ;

	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<IntWritable, MatrixBlock> out, Reporter reporter)
	throws IOException {
		inputConverter.setBlockSize(brlen, bclen); // 2 x 2 matrix blocks..
		inputConverter.convert(rawKey, rawValue);
		while(inputConverter.hasNext()) {
			Pair<MatrixIndexes, MatrixBlock> pair=inputConverter.next();
			block = pair.getValue() ;
			currRandom = new Well1024a();
			if(pp.idToStratify == -1)
				currRandom = random[0] ;
			else {
				domVal = (int) block.getValue(0, pp.idToStratify) ; //XXX change
				currRandom = random[domVal] ;
			}
			
			bmm.execute(currRandom, pair, reporter, out) ;
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
		numRows = MRJobConfiguration.getNumRows(job, (byte) 0);
		numColumns = MRJobConfiguration.getNumColumns(job, (byte) 0);
		pp = MRJobConfiguration.getPartitionParams(job);

		if(pp.isEL == false && pp.pt == PartitionParams.PartitionType.submatrix) {
			bmm = new SubMatrixBlockMapperMethod(pp, multipleOutputs, numRows, numColumns, brlen, bclen);
			random = new Well1024a[1] ; random[0] = new Well1024a() ;
		}
		else if(pp.isEL == true || (pp.isEL == false && pp.pt == PartitionParams.PartitionType.row)) {
			if(pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.kfold)
				bmm = new KFoldRowBlockMapperMethod(pp, multipleOutputs) ;
			else if((pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.holdout) || 
					(pp.isEL == true && (pp.et == PartitionParams.EnsembleType.rsm || pp.et == PartitionParams.EnsembleType.rowholdout)))
				bmm = new HoldoutRowBlockMapperMethod(pp, multipleOutputs) ;
			
			int mapperId = MapReduceTool.getUniqueMapperId(job, true) ;
			int numSeedsReqd = pp.getNumSeedsPerMapper() ;
			random = new Well1024a[numSeedsReqd] ;
			// Skip random[0] by numSeedsReqd * mapperId ;
			random[0] = new Well1024a() ;
			for(int i = 0 ; i < numSeedsReqd * mapperId; i++){
				// DOUG: RANDOM SUBSTREAM
				//random[0].resetNextSubstream() ;
			}
			for(int i = 1 ; i < random.length; i++) {
				// DOUG: RANDOM SUBSTREAM
				//random[i] = (Well1024a)random[i-1].clone() ;
				//random[i].resetNextSubstream() ;
			}
		}
	}
}
