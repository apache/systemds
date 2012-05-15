package com.ibm.bi.dml.meta;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.ListIterator;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.mapred.MRJobConfiguration;


public class PartitionBlockJoinReducer extends MapReduceBase 
implements Reducer<LongWritable, BlockJoinMapOutputValue, MatrixIndexes, MatrixValue>{
	protected MultipleOutputs multipleOutputs;
	PartitionParams pp = new PartitionParams() ;
	long dimension;
	@Override
	public void reduce(LongWritable key, Iterator<BlockJoinMapOutputValue> values,
			OutputCollector<MatrixIndexes, MatrixValue> out, Reporter reporter)
	throws IOException {

		//so, we'll get idtable pairs and cellvalues; we use cell values and fill them into a rowblk; later use ids to write to output files!
		MatrixBlock outblk = (pp.isColumn == false) ? new MatrixBlock(1, (int)dimension, true) :  
								new MatrixBlock((int)dimension, 1, true);	//assumed sparse; but cast to int can cause overflow!! TODO 
		ArrayList<Pair<Long, Integer>> joinlist = new ArrayList<Pair<Long, Integer>>(); //for btstrp/kfold/holdout
		while(values.hasNext()) {
			BlockJoinMapOutputValue val = new BlockJoinMapOutputValue(values.next());
			if(val.val2 == 0) {	//matrix element
				if(pp.isColumn == false)
					outblk.setValue(0, (int)val.val1, val.cellvalue);	//use absolcolid; cast to int can cause overflow!!! TODO
				else
					outblk.setValue((int)val.val1, 0, val.cellvalue);	//use absolcolid; cast to int can cause overflow!!! TODO
			}
			else if(val.val2 < 0) {	//holdout/kfold element
				joinlist.add(new Pair(new Long(val.val1), new Integer(-1*val.val2 - 1)));	//get bk correct foldnum encoded
			}
			else {	//bootstrap element
				joinlist.add(new Pair(new Long(val.val1), new Integer(val.val2 - 1)));	//get bk correct foldnum
			}
		}
		//now, we can write out the rowcolblk to resp fold files by joining them with futrowids
		if((pp.isEL == false && (pp.cvt == PartitionParams.CrossvalType.holdout || pp.cvt == PartitionParams.CrossvalType.kfold)) || 
				(pp.isEL == true && (pp.et == PartitionParams.EnsembleType.rsm || pp.et == PartitionParams.EnsembleType.rowholdout))) {
				//for(int i=0; i < joinlist.size(); i++) {
				ListIterator<Pair<Long, Integer>> jiter = joinlist.listIterator();
				while (jiter.hasNext()) {
					Pair <Long, Integer> apair = jiter.next();
					long outrowcolid = 0;	//need to rectify this in val1 based on replication and train/test
					int outfoldnum = 0;	//available from val2
					if((pp.isEL==false)&&(pp.cvt == PartitionParams.CrossvalType.kfold) && (pp.toReplicate == false)) {
						if(apair.getKey() > 0) {	//output only the k test folds	//TODO!! chck vla1 and output keyvalpair
							outfoldnum = apair.getValue();
							outrowcolid = apair.getKey() - 1;	//starts from 1 
						}
						else {	//train fold skipped; go to next entry in tuple
							continue;
						}
					}
					else if (pp.isEL == false) {	//cv kfold w repl; cv holdout row
						outfoldnum = (apair.getKey() > 0) ? 2*apair.getValue() : 2*apair.getValue() + 1;
						outrowcolid = (apair.getKey() > 0) ?  (apair.getKey() - 1)
											: (-1*apair.getKey()  - 1); //starts from 1 or -1
					}
					else if (pp.isEL == true) {		//el rsm or rowholdout
						if(apair.getKey() > 0)	//ignore test set
							continue;
						outfoldnum = apair.getValue();
						outrowcolid = -1 * apair.getKey() - 1;
					}
					else {
						System.out.println("Unrecognized method in join reducer!");
						System.exit(1);
					}
					MatrixIndexes indexes = (pp.isColumn == false) ? new MatrixIndexes(outrowcolid + 1, 1) :
									new MatrixIndexes(1, outrowcolid + 1) ;	//rowid/colid of subblk; systemml matrxblks from (1,1)
					reporter.incrCounter("counter", "" + outfoldnum, 1) ;
					multipleOutputs.getCollector("" + outfoldnum, reporter).collect(indexes, outblk) ;
				}
		}
		else if((pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.bootstrap) ||
				(pp.isEL == true && pp.et == PartitionParams.EnsembleType.bagging)) {	//has only train fold outputs
				//for (Pair<Long, Long> q : bslist) {		//wo repl, only single item
				ListIterator<Pair<Long, Integer>> jiter = joinlist.listIterator();
				while (jiter.hasNext()) {
					Pair <Long, Integer> apair = jiter.next();	
					MatrixIndexes indexes = new MatrixIndexes(apair.getKey() + 1, 1);	//rowid of subrowblk; systemml matrixblks from (1,1)
					reporter.incrCounter("counter", "" + apair.getValue(), 1) ;
					multipleOutputs.getCollector("" + apair.getValue(), reporter).collect(indexes, outblk) ;
				}
		}
		
		//this is the prev method of sending subrowblks around; instead now we send cell values around! probly more eff if v sparse!
		/*
		//effect the join between the subrowblocks/subcolblks and the futurerowid/colid, and write those out!
		ArrayList<Pair<Long, BlockJoinMapOutputValue>> blklist = new ArrayList<Pair<Long,BlockJoinMapOutputValue>>();
		ArrayList<Pair<Long, Integer>> joinlist = new ArrayList<Pair<Long, Integer>>(); //for btstrp/kfold/holdout
		//TODO: ARUN: xxxx recheck about maxrow vs rlen! examsparsity causes problems!!! xxxxx also matrix indexing 0 vs 1!
		while(values.hasNext()) {
			BlockJoinMapOutputValue val = new BlockJoinMapOutputValue(values.next());
			//val.blk.examSparsity();
			if(val.val2 == 0) {	//matrix element
				blklist.add(new Pair(new Long(val.val1), new BlockJoinMapOutputValue(val)));
			}
			else if(val.val2 < 0) {	//holdout/kfold element
				joinlist.add(new Pair(new Long(val.val1), new Integer(-1*val.val2 - 1)));	//get bk correct foldnum encoded
			}
			else {	//bootstrap element
				joinlist.add(new Pair(new Long(val.val1), new Integer(val.val2 - 1)));	//get bk correct foldnum
			}
		}
		//now, we can write out each subrowblk to resp fold files by joining subrowblks with futrowids
		if((pp.isEL == false && (pp.cvt == PartitionParams.CrossvalType.holdout || pp.cvt == PartitionParams.CrossvalType.kfold)) || 
				(pp.isEL == true && (pp.et == PartitionParams.EnsembleType.rsm || pp.et == PartitionParams.EnsembleType.rowholdout))) {
			for (Pair<Long, BlockJoinMapOutputValue> p : blklist) {
				//for(int i=0; i < joinlist.size(); i++) {
				ListIterator<Pair<Long, Integer>> jiter = joinlist.listIterator();
				while (jiter.hasNext()) {
					Pair <Long, Integer> apair = jiter.next();
					long outrowcolid = 0;	//need to rectify this in val1 based on replication and train/test
					int outfoldnum = 0;	//available from val2
					if((pp.isEL==false)&&(pp.cvt == PartitionParams.CrossvalType.kfold) && (pp.toReplicate == false)) {
						if(apair.getKey() > 0) {	//output only the k test folds	//TODO!! chck vla1 and output keyvalpair
							outfoldnum = apair.getValue();
							outrowcolid = apair.getKey() - 1;	//starts from 1 
						}
						else {	//train fold skipped; go to next entry in tuple
							continue;
						}
					}
					else if (pp.isEL == false) {	//cv kfold w repl; cv holdout row
						outfoldnum = (apair.getKey() > 0) ? 2*apair.getValue() : 2*apair.getValue() + 1;
						outrowcolid = (apair.getKey() > 0) ?  (apair.getKey() - 1)
											: (-1*apair.getKey()  - 1); //starts from 1 or -1
					}
					else if (pp.isEL == true) {		//el rsm or rowholdout
						if(apair.getKey() > 0)	//ignore test set
							continue;
						outfoldnum = apair.getValue();
						outrowcolid = -1 * apair.getKey() - 1;
					}
					else {
						System.out.println("Unrecognized method in join reducer!");
						System.exit(1);
					}
					MatrixIndexes indexes = (pp.isColumn == false) ? new MatrixIndexes(outrowcolid + 1, p.getKey() + 1) :
									new MatrixIndexes(p.getKey() + 1, outrowcolid + 1) ;	//rowid/colid of subblk; systemml matrxblks from (1,1)
					reporter.incrCounter("counter", "" + outfoldnum, 1) ;
					multipleOutputs.getCollector("" + outfoldnum, reporter).collect(indexes, p.getValue().blk) ;
				}
			}
		}
		else if((pp.isEL == false && pp.cvt == PartitionParams.CrossvalType.bootstrap) ||
				(pp.isEL == true && pp.et == PartitionParams.EnsembleType.bagging)) {	//has only train fold outputs
			for (Pair<Long, BlockJoinMapOutputValue> p : blklist) {
				//for (Pair<Long, Long> q : bslist) {		//wo repl, only single item
				ListIterator<Pair<Long, Integer>> jiter = joinlist.listIterator();
				while (jiter.hasNext()) {
					Pair <Long, Integer> apair = jiter.next();	
					MatrixIndexes indexes = new MatrixIndexes(apair.getKey() + 1, p.getKey() + 1);	//rowid of subrowblk; systemml matrixblks from (1,1)
					reporter.incrCounter("counter", "" + apair.getValue(), 1) ;
					multipleOutputs.getCollector("" + apair.getValue(), reporter).collect(indexes, p.getValue().blk) ;
				}
			}
		}*/
		
	}

	public void close() throws IOException {
		multipleOutputs.close();
	}
	@Override
	public void configure(JobConf job) {
		multipleOutputs = new MultipleOutputs(job);
		pp = MRJobConfiguration.getPartitionParams(job);
		dimension = (pp.isColumn == false) ? MRJobConfiguration.getNumColumns(job, (byte)0) : MRJobConfiguration.getNumRows(job, (byte)0); //1 inp!
	}
}