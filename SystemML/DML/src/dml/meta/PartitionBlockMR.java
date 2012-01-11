package dml.meta;

import java.io.IOException;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import umontreal.iro.lecuyer.rng.WELL1024;

import dml.runtime.matrix.JobReturn;
import dml.runtime.matrix.MatrixCharacteristics;
import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.InputInfo;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.OutputInfo;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.util.MapReduceTool;


public class PartitionBlockMR {
	public static JobReturn runJob(String input, InputInfo inputinfo, int numReducers, int replication,
			long nr, long nc, int bnr, int bnc, PartitionParams pp) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(PartitionBlockMR.class);
		job.setJobName("PartitionBlockMR");

if((pp.isEL == true && ((pp.et == PartitionParams.EnsembleType.rsm) || (pp.et == PartitionParams.EnsembleType.bagging))) ||
	(pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.bootstrap)){
	System.out.println("Reblock based partitioning should not arise for pp.isEL:"+pp.isEL+",pp.cvt:"+pp.cvt+",pp.et:"+pp.et+"!");
	System.exit(1);
}
		
		if(pp.isEL == false && pp.pt == PartitionParams.PartitionType.submatrix)
			pp.numFoldsForSubMatrix(nr, nc) ;
		String[] outputs = pp.getOutputStrings();
		byte[] resultIndexes = pp.getResultIndexes() ;
		byte[] resultDimsUnknown = pp.getResultDimsUnknown();
		MRJobConfiguration.setPartitionParams(job, pp) ;
		MRJobConfiguration.setUpMultipleInputs(job, new byte[]{0}, new String[]{input}, new InputInfo[]{inputinfo},
												true, new int[]{bnr}, new int[]{bnc});
		OutputInfo[] outputInfos = new OutputInfo[outputs.length] ;
		for(int i = 0 ; i < outputInfos.length; i++){
			outputInfos[i] = OutputInfo.BinaryBlockOutputInfo ;
			outputs[i] = "" + outputs[i];		//convert output varblname to filepathname
		}
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, true);

		job.setInt("dfs.replication", replication);
		MRJobConfiguration.setMatricesDimensions(job, new byte[]{0}, new long[]{nr}, new long[]{nc});
		MRJobConfiguration.setBlockSize(job, (byte)0, bnr, bnc);
		// configure mapper
		job.setMapperClass(PartitionBlockMapper.class);
		if(pp.isEL == true || (pp.isEL == false && pp.pt == PartitionParams.PartitionType.row))
			job.setMapOutputKeyClass(IntWritable.class) ;
		else	//assumed cv submatrix
			job.setMapOutputKeyClass(MatrixIndexes.class) ;
		job.setMapOutputValueClass(MatrixBlock.class);
		
		if(pp.isEL == false && pp.pt == PartitionParams.PartitionType.submatrix)
			job.setNumReduceTasks(0);
		else {	//assumed cv row
			job.setNumReduceTasks(numReducers) ;
			job.setReducerClass(PartitionBlockReducer.class) ;
			job.setOutputKeyClass(MatrixIndexes.class) ;
			job.setOutputValueClass(MatrixBlock.class) ;
		}

		/*job.setProfileEnabled(true) ;
		job.setProfileParams("-agentlib:hprof=cpu=samples,heap=sites,depth=10," +
		"force=n,thread=y,verbose=n,file=%s");*/

		for(int i = 0 ; i < outputs.length; i++) {
			Path outPath = new Path(outputs[i]) ;
			MapReduceTool.deleteFileIfExistOnHDFS(outPath, job) ;
		}

		JobClient jc = new JobClient(job) ;
		RunningJob rj = jc.runJob(job) ;

		if (pp.isEL == false && pp.pt == PartitionParams.PartitionType.submatrix) {
			Pair<long[],long[]> lengths = pp.getRowAndColumnLengths(nr, nc, bnr, bnc) ;
			MatrixCharacteristics[] mc = new MatrixCharacteristics[lengths.getKey().length] ;
			long[] rowArray = lengths.getKey() ; long[] colArray = lengths.getValue() ;
			for(int i = 0 ; i < mc.length; i++) {
				mc[i] = new MatrixCharacteristics(rowArray[i], colArray[i], bnr, bnc) ;
				System.out.println("DRB: (submatrix) " + outputs[i] + ": " + mc[i]) ;
			}
			return new JobReturn(mc, rj.isSuccessful());
		}
		//use reducer counters to ascertain output matrix characs
		else if((pp.isEL == false && pp.pt == PartitionParams.PartitionType.row) || 
						(pp.isEL == true)){		//cv all row / col methods; el all methods  
			Counters ctrs = rj.getCounters() ;
			// set 
			long[] ctr = null;
			
			if(pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.holdout) {
				if(pp.toReplicate == true)
					ctr = new long[pp.numIterations * 2] ;
				else
					ctr = new long[2] ;
			}
			else if(pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.kfold)  {
				if (pp.toReplicate == true)
					ctr = new long[pp.numFolds * 2] ;
				else
					ctr = new long[pp.numFolds] ;
			}
			else if(pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.bootstrap)  {
				if (pp.toReplicate == true)
					ctr = new long[pp.numIterations] ;
				else
					ctr = new long[1] ;
			}
			else if (pp.isEL == true) {
				if (pp.toReplicate == true)
					ctr = new long[pp.numIterations];
				else
					ctr = new long[1];
			}

			for(int i  = 0 ; i < ctr.length ; i++)
				ctr[i]  = ctrs.findCounter("counter", "" + i).getCounter() ;	//ctr gives total numrowcolblks perfold
			MatrixCharacteristics mc[] = new MatrixCharacteristics[ctr.length] ;
			for(int i = 0 ; i < mc.length; i++) {
				if(pp.isColumn == false) {	//we had subrowblks
					mc[i] = new MatrixCharacteristics(ctr[i], nc, 1, bnc);	//same num cols though
				}
				else {
					mc[i] = new MatrixCharacteristics(nr, ctr[i], bnr, 1);	//same num rows though
				}
				System.out.println("DRB: (row) " + outputs[i] + ": " + mc[i].toString());
			}
			return new JobReturn(mc, rj.isSuccessful()) ;
		}//end else if on row
		//else if (pp.pt == PartitionParams.PartitionType.cell) {
			//TODO
		//}
		return null;
	}
}