package dml.meta;

import java.io.File;
import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
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

//<Arun>
import java.util.HashMap;
import java.util.Random;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import java.net.URI;
import java.net.URISyntaxException;
//</Arun>
//this job is for reconstructing a train output column matrix, in case of el rsm (or cv col holdout?)
//also need to take issupervised into acnt
//NOTE: since we need the output matrix to also be in block format (1000x1000; though only 1 columns of it is occupied!),
//we have to do an entire Map+Reduce job! If we can write output in cell format, then Map only job suffices.
//For a partially occupied block, maxrow indicates max num rows actually present vs rlen which is blk dim in general
public class ReconstructionHashMapMR {
	public static JobReturn runJob(String input, InputInfo inputinfo, int numReducers, int replication,
				int bnr, int bnc, PartitionParams pp, int foldnum, String output)
	throws Exception
	{
		JobConf job;
		job = new JobConf(ReconstructionHashMapMR.class);
		job.setJobName("ReconstructionHashMapMR");
		
		String[] outputs = {output};	//only one output i.e. result matrix
		byte[] resultIndexes = {(byte)0};	
		byte[] resultDimsUnknown = {(byte)1};
		MRJobConfiguration.setPartitionParams(job, pp) ;
		MRJobConfiguration.setUpMultipleInputs(job, new byte[]{0}, new String[]{input}, new InputInfo[]{inputinfo},
												true, new int[]{bnr}, new int[]{bnc});
		OutputInfo[] outputInfos = new OutputInfo[outputs.length] ;
		for(int i = 0 ; i < outputInfos.length; i++){
			outputInfos[i] = OutputInfo.BinaryBlockOutputInfo ;
		}
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, true);

		job.setInt("dfs.replication", replication);
		job.setInt("foldnum", foldnum);
		MRJobConfiguration.setMatricesDimensions(job, new byte[]{0}, new long[]{-1}, new long[]{-1}); //dont know inp dims
		MRJobConfiguration.setBlockSize(job, (byte)0, bnr, bnc);	//assumed same blk size
		
		job.setMapperClass(ReconstructionHashMapMapper.class);
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(ReconstructionHashMapMapOutputValue.class);
		
		job.setNumReduceTasks(numReducers) ;
		job.setReducerClass(ReconstructionHashMapReducer.class) ;

		job.setOutputKeyClass(MatrixIndexes.class) ;
		job.setOutputValueClass(MatrixBlock.class) ;

		/*job.setProfileEnabled(true) ;
		job.setProfileParams("-agentlib:hprof=cpu=samples,heap=sites,depth=10," +
		"force=n,thread=y,verbose=n,file=%s");*/

		for(int i = 0 ; i < outputs.length; i++) {
			Path outPath = new Path(outputs[i]) ;
			MapReduceTool.deleteFileIfExistOnHDFS(outPath, job) ;
		}
		System.out.println("$$$$$$\t" +	"Running reconstruct job using hashmap!\t" + "$$$$$\n");
		JobClient jc = new JobClient(job) ;
		RunningJob rj = jc.runJob(job) ;

		Counters ctrs = rj.getCounters() ;
		long[] ctr = new long[1];	//only one output matrix
		for(int i  = 0 ; i < ctr.length ; i++)
			ctr[i]  = ctrs.findCounter("counter", "").getCounter() ;	//get the number of subrows from reducer
		MatrixCharacteristics mc[] = new MatrixCharacteristics[ctr.length] ;
		for(int i = 0 ; i < mc.length; i++) {
			mc[i] = new MatrixCharacteristics(ctr[i], 1, bnr, bnc) ;	//numrows, 1 col in orignl blk format
			System.out.println("DRB: (row) " + outputs[i] + ": " + mc[i].toString()) ;
		}
		return new JobReturn(mc, rj.isSuccessful()) ;
	}
}