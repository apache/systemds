/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;

import org.apache.commons.math.random.Well1024a;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.data.Converter;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class PartitionBlockMapper extends MapReduceBase
implements Mapper<Writable, Writable, IntWritable, MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
			
			int mapperId = MapReduceTool.getUniqueTaskId(job) ;
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
