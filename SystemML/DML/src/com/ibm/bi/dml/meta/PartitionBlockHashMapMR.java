package com.ibm.bi.dml.meta;

import java.net.URI;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;

import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;


public class PartitionBlockHashMapMR {
	public static JobReturn runJob(String input, InputInfo inputinfo, int numReducers, int replication,
			long nr, long nc, int bnr, int bnc, PartitionParams pp) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(PartitionBlockHashMapMR.class);
		job.setJobName("PartitionBlockHashMapMR");
		
		if(pp.pt == PartitionParams.PartitionType.submatrix)
			pp.numFoldsForSubMatrix(nr, nc) ;
		//String[] outputs = (pp.toReplicate) ? pp.getOutputStrings1() : pp.getOutputStrings2();
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
		long[] ctr = new long[outputs.length];	//used to collect output matr dimensions for row/col partitioning!

		job.setInt("dfs.replication", replication);
		MRJobConfiguration.setMatricesDimensions(job, new byte[]{0}, new long[]{nr}, new long[]{nc});
		MRJobConfiguration.setBlockSize(job, (byte)0, bnr, bnc);
		
		//thehashmap written to hdfs and used by mappers; all 3 types of partng; rowid key, diff value types
		//The new hashmap contains the actual rowid in the output matrix for each rowid in each fold (except btstrpng)
		//HashMap <Long, Long[]> thehashmap = new HashMap <Long, Long[]> ();
		//instead, initialize the hashmap with long[] arrays!
		VectorOfArrays thehashmap = null;	//we still use vectorofarrays here, and later when reading in, we use vectorofarraysbag
		Random rand = new Random(System.currentTimeMillis());
		
		//there are 3 cases: cv kfold (row), cv/el btstrp (row), cv holdout (row, unsup col, sup col) / el rsm/rowholdout
		//the other cases are not meaningful, eg  cv kfold col, btstrp col!! also, only pt=row uses this MRjob!
		if (pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.kfold) {
			Long[] testrunningcounts = new Long[pp.numFolds];	//used for future rowid determination
			Long[] trainrunningcounts = new Long[pp.numFolds];
			for(int j = 0; j < pp.numFolds; j++) {
				testrunningcounts[j] = new Long(1);
				trainrunningcounts[j] = new Long(1);	//we start from 1 on both since +/- distngshs test/train!
			}
			//for(long b=0; b < nr; b++)
			//	thehashmap.put(b, new Long[pp.numFolds]);
			thehashmap = new VectorOfArrays(nr, pp.numFolds);
			for(long i = 0; i < nr ; i++) {
				for(int b=0; b < pp.numFolds; b++)
					thehashmap.set(i, b, -1 * trainrunningcounts[b]++);
				int testfold = rand.nextInt(pp.numFolds);		//test in this fold, rest as train
				thehashmap.set(i,testfold, testrunningcounts[testfold]++);
				trainrunningcounts[testfold]--;		//discount this row from that fold's train matrix
			}
			//note down output matr dimensions!
			if(pp.toReplicate == false)
				for(int b=0; b < pp.numFolds; b++)
					ctr[b] = testrunningcounts[b] - 1;	//since we started from 1
			else
				for(int b=0; b < pp.numFolds; b++) {
					ctr[2*b] = testrunningcounts[b] - 1;	//since we started from 1
					ctr[2*b+1] = trainrunningcounts[b] - 1;	//since we started from 1
				}
		}
		else if ((pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.bootstrap) ||
						(pp.isEL == true && pp.et == PartitionParams.EnsembleType.bagging)){
			int numtimes = pp.numIterations;	//no replication -> write only once
			if(pp.toReplicate == false)
				numtimes = 1;
			/*//for now, i only retain the counts of the occurrences in the folds 
			long numsamples = Math.round(pp.frac * nr);		//num btstrp smpls
			for(long b=0; b < nr; b++) {
				thehashmap.put(b, new Long[numtimes]);
				//initlze counts
				for(int j = 0; j < numtimes; j++) {
					thehashmap.get(b)[j] = new Long(0);
				}
			}
			for(int j = 0; j < numtimes; j++) {
				for(long i = 0; i < numsamples; i++) {
					long chosen = rand.nextLong() % nr;			//unif rand from 0 to nr-1 - sample w repl
					thehashmap.get(chosen)[j]++;
				}
			}*/ //the prev hm frmat for btstrp (using occ counts) instead, ive reverted to jr formt for btstrp as below! (swap-based)
			Random[] randi = new Random[numtimes];	//numtimes pnrg for samplings with replacement
			for(int j = 0; j < numtimes; j++) {
				randi[j] = new Random(rand.nextInt());	//each is randomly seeded - is this ok? TODO
			}
			//for join, a separate hashmap!! -> with value-key swap on idtable!
			long numsamples = Math.round(pp.frac * nr);		//num btstrp smpls
			System.out.println("$$$$$$$$ In partnblkhashmapMR, btstrp/baggin has numsamples = " + pp.frac + "*" + nr + "=" + numsamples);
			thehashmap = new VectorOfArrays(numsamples, numtimes);
			for(long i = 0; i < numsamples; i++) {
				//thehashmap.put(i, new Long[numtimes]);
				for(int j = 0; j < numtimes; j++) {
					thehashmap.set(i, j, (long)randi[j].nextInt((int)nr));	//TODO: the curr rowid in sample for later; can overflows!!! ###
					//thehashmap.get(i)[j] = (long)randi[j].nextInt((int)nr);	//TODO: the curr rowid in sample for later; can overflows!!! ###
				}
			}
			//note down output matr dims
			for(int b=0; b < numtimes; b++)
				ctr[b] = numsamples;	//same across folds!
		}
		//cv holdout or el rsm/rowholdout; cv holdout can be row or unsup column or sup column; el rsm is implicitly sup col
		else if ((pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.holdout) || 
					(pp.isEL == true && (pp.et == PartitionParams.EnsembleType.rsm || pp.et == PartitionParams.EnsembleType.rowholdout))) {
			int numtimes = pp.numIterations;	//no replication -> write only once
			if(pp.toReplicate == false)
				numtimes = 1;
			Long[] testrunningcounts = new Long[numtimes];	//used for future rowid determination
			Long[] trainrunningcounts = new Long[numtimes];
			for(int j = 0; j < numtimes; j++) {
				testrunningcounts[j] = new Long(1);
				trainrunningcounts[j] = new Long(1);	//we start from 1 on both since +/- distngshs test/train!
			}
			long nk;
			if(pp.isEL == false || (pp.isEL == true && pp.et == PartitionParams.EnsembleType.rowholdout)) { //cv or el rowholdout
				/*if (pp.isColumn == false) //row
					nk = nr;
				else if (pp.isSupervised == false) //unsup col
					nk = nc;
				else
					nk = nc - 1;	//sup col*/ //cv col holdout doesnt make sense either
				nk = nr;
			}
			else	//el rsm
				nk = nc - 1;
			//for(long b=0; b < nk; b++)
			//	thehashmap.put(b, new Long[numtimes]);
			if(pp.isEL == true && pp.et == PartitionParams.EnsembleType.rsm)
				thehashmap = new VectorOfArrays(nk + 1, numtimes);	//incl last entry for labels
			else
				thehashmap = new VectorOfArrays(nk, numtimes);
			for(long i = 0; i < nk ; i++) {
				for(int j = 0; j < numtimes; j++) {
					double value = rand.nextDouble();
					//based on frac, we set index -> + means test's index, - means train's index
					thehashmap.set(i,j, (value < pp.frac) ? 
							testrunningcounts[j]++ : -1 * trainrunningcounts[j]++);
				}
			}
			//if it is el rsm, final col should go to all train folds as last entry!
			//if(pp.isEL == true || (pp.isEL == false && pp.isColumn == true && pp.isSupervised == true)) {
			//col sampling makes sense only for el rsm
			if(pp.isEL == true && pp.et == PartitionParams.EnsembleType.rsm) {
				//thehashmap.put(nc-1, new Long[numtimes]);
				for(int j = 0; j < numtimes; j++) {
					thehashmap.set((nc - 1), j, trainrunningcounts[j]++);	//final col
				}
			}
			//note down output matr dimensions!
			for(int b=0; b < numtimes; b++) {
				if(pp.isEL == false) {	//for cv, both test and train folds are output
					ctr[2*b] = testrunningcounts[b] - 1;	//since we started from 1
					ctr[2*b+1] = trainrunningcounts[b] - 1;	//since we started from 1
				}
				else	//for el, only train folds are output
					ctr[b] = trainrunningcounts[b] - 1;	//since we started from 1
			}
		}
		else {
			System.out.println("Unsupported method in hashmap MR job!");
			System.exit(1);
		}
		//write out the hashmap to a seqfile
		String uri = new String(pp.sfmapfile);
		Path sfpath = new Path(uri);
		Configuration sfconf = new Configuration();
		sfconf.set("CompressionType", "BLOCK");
		FileSystem fs = FileSystem.get(URI.create(uri),sfconf);
		LongWritable key = new LongWritable();
		WritableLongArray value = new WritableLongArray(thehashmap.width);
		SequenceFile.Writer sfwriter = new SequenceFile.Writer(fs, sfconf, sfpath, LongWritable.class, WritableLongArray.class);
		long N = thehashmap.length;
		
		for(long i = 0; i < N ; i++) {
			key.set(i);
			for(int j=0; j<thehashmap.width; j++)
				value.array[j] = thehashmap.get(i,j);
			//value.set(thehashmap.width, thehashmap.get(i));
			//System.out.println("$$$$$$ In partnblkhashmpMR, writing hashmap tuple key: " + key.toString() + ", value: " + value.toString());
			sfwriter.append(key, value);
		}
		sfwriter.close();
		//System.out.println("$$$$$$ Written out hashmap to seq file! $$$$$$$");

		DistributedCache.addCacheFile(URI.create(uri), job);	//copy mapfile to distr cache!
		//System.out.println("In partnblkhshmapMR, after set up of distr cache, job(mapred.cachce.files) is " + job.get("mapred.cache.files"));		
		
		job.setMapperClass(PartitionBlockHashMapMapper.class);
		job.setMapOutputKeyClass(BlockHashMapMapOutputKey.class);
		job.setMapOutputValueClass(BlockHashMapMapOutputValue.class);
		
		job.setNumReduceTasks(numReducers) ;
		job.setReducerClass(PartitionBlockHashMapReducer.class) ;

		job.setOutputKeyClass(MatrixIndexes.class) ;
		job.setOutputValueClass(MatrixBlock.class) ;

		/*job.setProfileEnabled(true) ;
		job.setProfileParams("-agentlib:hprof=cpu=samples,heap=sites,depth=10," +
		"force=n,thread=y,verbose=n,file=%s");*/

		for(int i = 0 ; i < outputs.length; i++) {
			Path outPath = new Path(outputs[i]) ;
			MapReduceTool.deleteFileIfExistOnHDFS(outPath, job) ;
		}
		//System.out.println("$$$$$$\t" +	"Running partition job!\t" + "$$$$$\n");
		JobClient jc = new JobClient(job) ;
		RunningJob rj = jc.runJob(job) ;

		if (pp.pt == PartitionParams.PartitionType.submatrix) {
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
			//prevly, we got output matrix dimensions by using reducer counters; now, since we use ReBlock-style trick of sending
			//only non-empty blk, that wont work! so, instead, we obtained the ctr[] from the hashmap itself!!
			MatrixCharacteristics mc[] = new MatrixCharacteristics[ctr.length] ;
			for(int i = 0 ; i < mc.length; i++) {
				if(pp.isColumn == false)
					mc[i] = new MatrixCharacteristics(ctr[i], nc, bnr, bnc);	//same num cols though
				else
					mc[i] = new MatrixCharacteristics(nr, ctr[i], bnr, bnc);	//same num rows though
				System.out.println("DRB: (row) " + outputs[i] + ": " + mc[i].toString() + " with ctr from reducer = " + 
						rj.getCounters().findCounter("counter", "" + i).getCounter());
			}
			/*Counters ctrs = rj.getCounters() ;
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
			long numblksx = (nc % bnc) == 0 ? nc / bnc : nc / bnc + 1; //num blks horizontally in orig input
			long numblksy = (nr % bnr) == 0 ? nr / bnr : nr / bnr + 1; //vertically; colwise output needs it

			for(int i  = 0 ; i < ctr.length ; i++)
				ctr[i]  = ctrs.findCounter("counter", "" + i).getCounter() ;	//ctr gives total numsubrowcolblks perfold
			MatrixCharacteristics mc[] = new MatrixCharacteristics[ctr.length] ;
			for(int i = 0 ; i < mc.length; i++) {
				if(pp.isColumn == false) {	//we had subrowblks
					if((ctr[i] % numblksx) != 0) {
						System.out.println("Error! Number of subrowblocks don't tally in blockhashmap reducer! !");
						System.exit(1);
					}
					mc[i] = new MatrixCharacteristics(ctr[i] / numblksx, nc, bnr, bnc);	//same num cols though
				}
				else {
					if((ctr[i] % numblksy) != 0) {
						System.out.println("Error! Number of subcolblocks don't tally in blockhashmap reducer! !");
						System.exit(1);
					}
					mc[i] = new MatrixCharacteristics(nr, ctr[i] / numblksy, bnr, bnc);	//same num rows though
				}
				System.out.println("DRB: (row) " + outputs[i] + ": " + mc[i].toString());
			}*/
			return new JobReturn(mc, rj.isSuccessful()) ;
		}//end else if on row
		//else if (pp.pt == PartitionParams.PartitionType.cell) {
			//TODO
		//}
		return null;
	}
}