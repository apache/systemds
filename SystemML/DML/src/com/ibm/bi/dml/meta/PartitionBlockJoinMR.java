/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.net.URI;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.lib.MultipleInputs;

import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;
import com.ibm.bi.dml.runtime.util.MapReduceTool;

//</Arun>
//TODO: change the driver for join!
public class PartitionBlockJoinMR 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static JobReturn runJob(String input, InputInfo inputinfo, int numReducers, int replication,
			long nr, long nc, int bnr, int bnc, PartitionParams pp) 
	throws Exception
	{
		JobConf job;
		job = new JobConf(PartitionBlockJoinMR.class);
		job.setJobName("PartitionBlockJoinMR");
		
		if(pp.pt == PartitionParams.PartitionType.submatrix)
			pp.numFoldsForSubMatrix(nr, nc);
		MRJobConfiguration.setPartitionParams(job, pp) ;
		//String[] outputs = (pp.toReplicate) ? pp.getOutputStrings1() : pp.getOutputStrings2();
		String[] outputs = pp.getOutputStrings();
		byte[] resultIndexes = pp.getResultIndexes();
		byte[] resultDimsUnknown = pp.getResultDimsUnknown();
		OutputInfo[] outputInfos = new OutputInfo[outputs.length] ;
		for(int i = 0 ; i < outputInfos.length; i++){
			outputInfos[i] = OutputInfo.BinaryBlockOutputInfo ;
			outputs[i] = "" + outputs[i];		//convert output varblname to filepathname
		}
		MRJobConfiguration.setUpMultipleOutputs(job, resultIndexes, resultDimsUnknown, outputs, outputInfos, true);
		long[] ctr = new long[outputs.length];	//used to collect output matr dimensions for row/col partitioning!

		job.setInt("dfs.replication", replication);
//shirish: prob need new InputInfo definition, incl key class and vlaue class; chk with ytain 
//ytain: use the new stmt for both inputs! remove the above job addition for mapper class!
//finally, we also had to import hadoop rather than hadoopfix. 
		//MRJobConfiguration.setUpMultipleInputs(job, new byte[]{0}, new String[]{input}, new InputInfo[]{inputinfo},
		//true, new int[]{bnr}, new int[]{bnc});//set up names of the input matrices and their inputformat information
		//job.setMapperClass(PartitionBlockJoinMapperMatrix.class);
		//instead of above stmt, we cull out the foll from that file
		job.setStrings("input.matrices.dirs", new String[]{input});		//TODO the prefix string is hardcoded from MRJobConfiguration.java
		MRJobConfiguration.setMapFucInputMatrixIndexes(job, new byte[]{0});
		MRJobConfiguration.setInputInfo(job, (byte)(0), inputinfo, true, bnr, bnc, false);
		//add idtable class
		MultipleInputs.addInputPath(job, new Path(input), inputinfo.inputFormatClass ,PartitionBlockJoinMapperMatrix.class); 

		MRJobConfiguration.setMatricesDimensions(job, new byte[]{0}, new long[]{nr}, new long[]{nc});
		MRJobConfiguration.setBlockSize(job, (byte)0, bnr, bnc);
	
		//add the idtable file to inputs!
		String uri = new String(pp.sfmapfile);
		Path sfpath = new Path(uri);
		MultipleInputs.addInputPath(job, sfpath, SequenceFileInputFormat.class, PartitionBlockJoinMapperIDTable.class); 

		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(BlockJoinMapOutputValue.class);

		//The new hashmap contains the actual rowid in the output matrix for each rowid in each fold (except btstrpng)
		//since we dont have space in memory for full hashmap, we maintain only the ith tuple entry
		Long[] thehashmapi = null;
		Random rand = new Random(System.currentTimeMillis());

		Configuration sfconf = new Configuration();
		sfconf.set("CompressionType", "BLOCK");
		FileSystem fs = FileSystem.get(URI.create(uri),sfconf);
		//LocalFileSystem fs = FileSystem.getLocal(sfconf);
		LongWritable key = new LongWritable();
		WritableLongArray value = new WritableLongArray();
		SequenceFile.Writer sfwriter = new SequenceFile.Writer(fs, sfconf, sfpath, 
				LongWritable.class, WritableLongArray.class);

		//there are 3 cases: cv kfold (row), cv/el btstrp (row), cv holdout (row, unsup col, sup col) / el rsm/rowholdout
		//the other cases are not meaningful, eg  cv kfold col, btstrp col!! also, only pt=row uses this MRjob!
		if (pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.kfold) {
			Long[] testrunningcounts = new Long[pp.numFolds];	//used for future rowid determination
			Long[] trainrunningcounts = new Long[pp.numFolds];
			for(int j = 0; j < pp.numFolds; j++) {
				testrunningcounts[j] = new Long(1);
				trainrunningcounts[j] = new Long(1);	//we start from 1 on both since +/- distngshs test/train!
			}
			for(long i = 0; i < nr ; i++) {
				thehashmapi = new Long[pp.numFolds];
				for(int b=0; b < pp.numFolds; b++)
					thehashmapi[b] = -1 * trainrunningcounts[b]++;
				int testfold = rand.nextInt(pp.numFolds);		//test in this fold, rest as train
				thehashmapi[testfold] = testrunningcounts[testfold]++;
				trainrunningcounts[testfold]--;		//discount this row from that fold's train matrix
				key.set(i);
				value.set(thehashmapi.length, thehashmapi);
				sfwriter.append(key, value);
				//System.out.println("$$$$$$$$$ Adding to idtable key:"+key.toString()+",value:"+value.toString());
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
			Random[] randi = new Random[numtimes];	//numtimes pnrg for samplings with replacement
			for(int j = 0; j < numtimes; j++) {
				randi[j] = new Random(rand.nextInt());	//each is randomly seeded - is this ok? TODO
			}
			//for join, a separate hashmap!! -> with value-key swap on idtable!
			long numsamples = Math.round(pp.frac * nr);		//num btstrp smpls
			for(long i = 0; i < numsamples; i++) {
				thehashmapi = new Long[numtimes];
				for(int j = 0; j < numtimes; j++) {
					thehashmapi[j] = (long)randi[j].nextInt((int)nr);	//TODO: the curr rowid in sample for later; can cause ###### overflows!!!
				}
				key.set(i);
				value.set(thehashmapi.length, thehashmapi);
				sfwriter.append(key, value);
				//System.out.println("$$$$$$$$$ Adding to idtable key:"+key.toString()+",value:"+value.toString());
			}
			//note down output matr dims
			for(int b=0; b < numtimes; b++)
				ctr[b] = numsamples;	//same across folds!
		}
		//cv holdout or el rsm/rowholdout; cv holdout can be row or unsup column or sup column; el rsm is implicitly sup col
		else if ((pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.holdout) || 
					(pp.isEL == true && (pp.et == PartitionParams.EnsembleType.rsm || pp.et == PartitionParams.EnsembleType.rowholdout))) {
			Long[] testrunningcounts = new Long[pp.numIterations];	//used for future rowid determination
			Long[] trainrunningcounts = new Long[pp.numIterations];
			int numtimes = pp.numIterations;	//no replication -> write only once
			if(pp.toReplicate == false)
				numtimes = 1;
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
					nk = nc - 1;	//sup col*/ //for cv, col doesnt make sense at all
				nk = nr;
			}
			else	//el rsm
				nk = nc - 1;
			for(long i = 0; i < nk ; i++) {
				thehashmapi = new Long[numtimes];
				for(int j = 0; j < numtimes; j++) {
					double randval = rand.nextDouble();
					thehashmapi[j] = (randval < pp.frac) ? 
								testrunningcounts[j]++ : -1 * trainrunningcounts[j]++;
				}
				key.set(i);
				value.set(thehashmapi.length, thehashmapi);
				sfwriter.append(key, value);
				//System.out.println("$$$$$$$$$ Adding to idtable key:"+key.toString()+",value:"+value.toString());
			}
			//if it is el rsm or cv sup col, final col should go to all train folds as last entry!
			//if(pp.isEL == true || (pp.isEL == false && pp.isColumn == true && pp.isSupervised == true)) {
			//col makes sense only for el rsm
			if(pp.isEL == true && pp.et == PartitionParams.EnsembleType.rsm) {
				thehashmapi = new Long[numtimes];
				for(int j = 0; j < numtimes; j++) {
					thehashmapi[j] = trainrunningcounts[j]++;
				}
				key.set(nc - 1);	//final col
				value.set(thehashmapi.length, thehashmapi);
				sfwriter.append(key, value);
				//System.out.println("$$$$$$$$$ Adding to idtable key:"+key.toString()+",value:"+value.toString());
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
			System.out.println("Unsupported method in join MR job!");
			System.exit(1);
		}

		sfwriter.close();
		System.out.println("$$$$$$$$$ Written out idtable seqfile! $$$$$$$$\n");

	    /*String inputFormatMapping = sfpath.toString() + ";"
	       + SequenceFileInputFormat.class.getName();
	    String inputFormats = job.get("mapred.input.dir.formats");
	    job.set("mapred.input.dir.formats",
	       inputFormats == null ? inputFormatMapping : inputFormats + ","
	           + inputFormatMapping);

	    String mapperMapping = sfpath.toString() + ";" + PartitionBlockJoinMapperIDTable.class.getName();
	    String mappers = job.get("mapred.input.dir.mappers");
	    job.set("mapred.input.dir.mappers", mappers == null ? mapperMapping
	       : mappers + "," + mapperMapping);*/

				
				
		job.setNumReduceTasks(numReducers) ;
		job.setReducerClass(PartitionBlockJoinReducer.class) ;
		job.setOutputKeyClass(MatrixIndexes.class) ;
		job.setOutputValueClass(MatrixBlock.class) ;

		/*job.setProfileEnabled(true) ;
		job.setProfileParams("-agentlib:hprof=cpu=samples,heap=sites,depth=10," +
		"force=n,thread=y,verbose=n,file=%s");*/

		for(int i = 0 ; i < outputs.length; i++) {
			Path outPath = new Path(outputs[i]) ;
			MapReduceTool.deleteFileIfExistOnHDFS(outPath, job) ;
		}

		JobClient jc = new JobClient(job) ;
		RunningJob rj = jc.runJob(job) ;

		if(pp.pt == PartitionParams.PartitionType.submatrix) {
			Pair<long[],long[]> lengths = pp.getRowAndColumnLengths(nr, nc, bnr, bnc) ;
			MatrixCharacteristics[] mc = new MatrixCharacteristics[lengths.getKey().length] ;
			long[] rowArray = lengths.getKey() ; long[] colArray = lengths.getValue() ;
			for(int i = 0 ; i < mc.length; i++) {
				mc[i] = new MatrixCharacteristics(rowArray[i], colArray[i], bnr, bnc) ;
				System.out.println("DRB: " + outputs[i] + ": " + mc[i]) ;
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
					mc[i] = new MatrixCharacteristics(ctr[i] / numblksx, nc, 1, bnc);	//same num cols though; rowblk o/p
				}
				else {
					if((ctr[i] % numblksy) != 0) {
						System.out.println("Error! Number of subcolblocks don't tally in blockhashmap reducer! !");
						System.exit(1);
					}
					mc[i] = new MatrixCharacteristics(nr, ctr[i] / numblksy, bnr, 1);	//same num rows; colblk o/p
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