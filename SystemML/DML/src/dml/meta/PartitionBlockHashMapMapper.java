package dml.meta;

import java.io.IOException;
import java.util.Vector;

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

public class PartitionBlockHashMapMapper extends MapReduceBase
implements Mapper<Writable, Writable, BlockHashMapMapOutputKey, BlockHashMapMapOutputValue> {
	
	private Converter inputConverter=null;
	PartitionParams pp = new PartitionParams() ;
	int brlen, bclen ;
	BlockHashMapMapperMethod bmm ;
	MultipleOutputs multipleOutputs ;

	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<BlockHashMapMapOutputKey, BlockHashMapMapOutputValue> out, Reporter reporter)
	throws IOException {
		inputConverter.setBlockSize(brlen, bclen);
		inputConverter.convert(rawKey, rawValue);
		while(inputConverter.hasNext()) {
			Pair<MatrixIndexes, MatrixBlock> pair=inputConverter.next();
			bmm.execute(pair, reporter, out) ;
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
		long hmlength = (pp.isColumn == true) ? MRJobConfiguration.getNumColumns(job, (byte)0) : 
			MRJobConfiguration.getNumRows(job, (byte)0); //1 inp! we get the no.of entries in the hashmpa, and compute size
		int hmwidth=1;
		//System.out.println("In partnblkhshmapMapper, job(mapred.cachce.files) is " + job.get("mapred.cache.files"));		
		
		//only row partitioning using hashmap
		if(pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.kfold) {
			bmm = new KFoldBlockHashMapMapperMethod(pp, multipleOutputs) ;
			hmwidth = pp.numFolds;
		}
		else if((pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.holdout) || 
				(pp.isEL == true && (pp.et == PartitionParams.EnsembleType.rsm || pp.et == PartitionParams.EnsembleType.rowholdout))) {
			bmm = new HoldoutBlockHashMapMapperMethod(pp, multipleOutputs) ;
			hmwidth = (pp.toReplicate == true) ? pp.numIterations : 1;
		}
		else if((pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.bootstrap) ||
				(pp.isEL == true && pp.et == PartitionParams.EnsembleType.bagging)) {
			bmm = new BootstrapBlockHashMapMapperMethod(pp, multipleOutputs) ;
			//hmlength = Math.round(hmlength * pp.frac); //actl num samples used in partnblkhshmapMR - for vectofarr; but now, bags, so frac sep!
			hmwidth = (pp.toReplicate == true) ? pp.numIterations : 1;
		}
		else {
			System.out.println("Unknown methods in partnblkhshmpmapper configure!" +
					"pp.isEL:" +pp.isEL + ",pp.cvt:"+pp.cvt+"pp.pt:"+pp.pt+",pp.et:"+pp.et);
			System.exit(1);
		}
		//read the hashmap into memory
		try {
			bmm.sfReadHashMap(job, hmlength, hmwidth);
		} catch (IOException e) {
			System.out.println("Could not read hashmap into memory!");
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		} catch (InstantiationException e) {
			e.printStackTrace();
		}
	}
}
