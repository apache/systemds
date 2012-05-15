package com.ibm.bi.dml.meta;

import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.RunningJob;

import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.io.PartialBlock;
import com.ibm.bi.dml.runtime.matrix.io.TaggedFirstSecondIndexes;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class PartitionSubMatrixMR {
	public static JobReturn runJob(boolean inBlockRepresentation,
			String input, InputInfo inputinfo, int numReducers, int replication,
			long nr, long nc, int bnr, int bnc, PartitionParams pp) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(PartitionSubMatrixMR.class);
		job.setJobName("PartitionSubMatrixMR");
		
		// first difference
		pp.numFoldsForSubMatrix(nr, nc) ;
		
		// set the names of the outputs
		String[] outputs = pp.getOutputStrings();
		byte[] resultIndexes = pp.getResultIndexes() ;
		byte[] resultDimsUnknown = pp.getResultDimsUnknown();
		
		MRJobConfiguration.setPartitionParams(job, pp) ;
		
		MRJobConfiguration.setUpMultipleInputs(job, new byte[]{0}, new String[]{input}, new InputInfo[]{inputinfo},
											   false, new int[]{bnr}, new int[]{bnc});
		
		OutputInfo[] outputInfos = new OutputInfo[outputs.length] ;
		for(int i = 0 ; i < outputInfos.length; i++)
			outputInfos[i] = OutputInfo.BinaryBlockOutputInfo ;
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, true);
		
		job.setNumReduceTasks(numReducers);
		job.setInt("dfs.replication", replication);
		
		MRJobConfiguration.setMatricesDimensions(job, new byte[]{0}, new long[]{nr}, new long[]{nc});
		MRJobConfiguration.setBlockSize(job, (byte)0, bnr, bnc);
		
		// configure mapper
		job.setMapperClass(PartitionSubMatrixMapperCell.class);
		job.setMapOutputKeyClass(TaggedFirstSecondIndexes.class);
		job.setMapOutputValueClass(PartialBlock.class);
		
		//configure reducer
		job.setReducerClass(PartitionSubMatrixReducerCell.class);
		job.setOutputKeyClass(MatrixIndexes.class);
		job.setOutputValueClass(MatrixBlock.class) ;
		job.setPartitionerClass(CustomPartitioner.class) ;
	
		JobClient jc = new JobClient(job) ;
		RunningJob rj=jc.runJob(job) ;
		
		Pair<long[],long[]> lengths = pp.getRowAndColumnLengths(nr, nc) ;
		MatrixCharacteristics[] mc = new MatrixCharacteristics[lengths.getKey().length] ;
		long[] rowArray = lengths.getKey() ; long[] colArray = lengths.getValue() ;
		for(int i = 0 ; i < mc.length; i++){
			mc[i] = new MatrixCharacteristics(rowArray[i], colArray[i], bnr, bnc) ;
			mc[i].nonZeros = rowArray[i] * colArray[i];
		}
		return new JobReturn(mc, rj.isSuccessful()) ;
	}
	
	static class CustomPartitioner implements Partitioner<TaggedFirstSecondIndexes, PartialBlock>{
	    @Override
	    public int getPartition(TaggedFirstSecondIndexes key, PartialBlock value, 
	                            int numPartitions) {
	      return new Long(key.getTag()*127).hashCode()%10007%numPartitions;
	    }

		@Override
		public void configure(JobConf arg0) {
			
		}
	}
}
