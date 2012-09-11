package com.ibm.bi.dml.meta;
//<Arun>
import java.io.IOException;

import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.IJV;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM.SparseCellIterator;


public class KFoldBlockHashMapMapperMethod extends BlockHashMapMapperMethod {

	public KFoldBlockHashMapMapperMethod(PartitionParams pp,
			MultipleOutputs multipleOutputs) {
		super(pp, multipleOutputs);
	}

	//Get subrows; send to reducer with map output key <foldid, (blkx, blky)> and  subrowid, matrixblk in value 
	@Override
	void execute(Pair<MatrixIndexes, MatrixBlock> pair, Reporter reporter, OutputCollector out)	throws IOException {
		long blky = pair.getKey().getRowIndex() - 1;
		long blkx = pair.getKey().getColumnIndex() - 1;	//since systemml uses blkindices from (1,1)
		int rpb = pp.rows_in_block;		//TODO: assuming this is the general block y dimension
		long N = thehashmap.length;
		
		///*
		//this is the even-more-eff cell kv pairs version! 1/c vs sparsity decision.
		MatrixBlock thisblock = pair.getValue();
		int nrows = thisblock.getNumRows();
		int ncols = thisblock.getNumColumns();
		int nk = (pp.isColumn == true) ? ncols : nrows;
		int nother = (pp.isColumn == false) ? ncols : nrows;
		BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
		key.blkx = blkx;	//the x index is preserved
		BlockHashMapMapOutputValue value = new BlockHashMapMapOutputValue();
		boolean issparse = thisblock.isInSparseFormat();
		if( issparse ) {
			SparseCellIterator iter = thisblock.getSparseCellIterator();
			while(iter.hasNext()) {
				IJV e = iter.next();
				value.cellvalue = e.v;
				value.locator = e.j;
				if(pp.toReplicate) {
					for(int i = 0; i < pp.numFolds ; i++) {
						//long entry = thehashmap.get(blky * rpb + e.getKey().row, i);
						long entry = thehashmap.get(blky * rpb + e.i, i).get(0);	//retvals is a single entry vector
						if(entry > 0) {		//the entry is the row index in the test output matrix of fold
							entry--;	//since we started from 1
							key.foldid = 2*i;
						}
						else {
							entry = -1*entry - 1;	//since we started from -1
							key.foldid = 2*i + 1;
						}
						value.auxdata = (int) (entry % rpb);	//TODO: could this give wrong results? 
						//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
						key.blky = entry / rpb;	//casting as long itself does flooring
						out.collect(key, value);
					}
				}
				else {	//no replication, output only k test fold entries!
					for(int i = 0; i < pp.numFolds ; i++) {
						long entry = thehashmap.get(blky * rpb + e.i, i).get(0);
						if(entry > 0) {		//the entry is the row index in the test output matrix of fold
							value.auxdata = (int) (entry % rpb);	//TODO: could this give wrong results?
							//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
							key.blky = entry / rpb;
							key.foldid = i;
							key.blkx = blkx;	//the x index is preserved
							out.collect(key, value);
						}
					}
				}
			}
		}
		else {
			double[] darray = thisblock.getDenseArray();
			if (darray != null) {
				long limit = nrows * ncols;
				for(int dd=0; dd<limit; dd++) {
					if(darray[dd] == 0)
						continue;
					int subrowid = dd / ncols;		//we get the row id within the block
					int subcolid = dd - subrowid*ncols;	//darray is in row-major form
					value.cellvalue = darray[dd];
					value.locator = subcolid;
					if(pp.toReplicate) {
						for(int i = 0; i < pp.numFolds ; i++) {
							long entry = thehashmap.get(blky * rpb + subrowid, i).get(0);
							if(entry > 0) {		//the entry is the row index in the test output matrix of fold
								entry--;	//since we started from 1
								key.foldid = 2*i;
							}
							else {
								entry = -1*entry - 1;	//since we started from -1
								key.foldid = 2*i + 1;
							}
							value.auxdata = (int) (entry % rpb);	//TODO: could this give wrong results? 
							//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
							key.blky = entry / rpb;	//casting as long itself does flooring
							out.collect(key, value);
						}
					}
					else {	//no replication, output only k test fold entries!
						for(int i = 0; i < pp.numFolds ; i++) {
							long entry = thehashmap.get(blky * rpb + subrowid, i).get(0);
							if(entry > 0) {		//the entry is the row index in the test output matrix of fold
								value.auxdata = (int) (entry % rpb);	//TODO: could this give wrong results?
								//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
								key.blky = entry / rpb;
								key.foldid = i;
								key.blkx = blkx;	//the x index is preserved
								out.collect(key, value);
							}
						}
					}
				}
			}
		}
		//*/
		
		/*
		//both hm/jr can send out cell kv pairs like ReBlock -> the cost competition is 1/c vs sparsity
		//for very sparse (<0.1%), cell kv pairs might be more efficient since mn/c vs mns => 1/c=0.001 > s!! TODO for now, ignored!
		//we send out subrowblks, but populate them by checking if this blk is dense/sparse! this requires buffering all subblks in mem!
		MatrixBlock thisblock = pair.getValue();
		int nrows = thisblock.getNumRows();
		int ncols = thisblock.getNumColumns();
		//MatrixBlock[] subblocks = new MatrixBlock[nrows];
		BlockHashMapMapOutputValue[] values = new BlockHashMapMapOutputValue[nrows]; 
		for(int r=0; r < nrows; r++) {
			values[r] = new BlockHashMapMapOutputValue();
			values[r].blk = new MatrixBlock(1, ncols, true);	//presume sparse!
		}
		boolean issparse = thisblock.isInSparseFormat();
		if((issparse == true) && (thisblock.getSparseMap() != null)) {
			Iterator<Entry<CellIndex, Double>> iter = thisblock.getSparseMap().entrySet().iterator();
			while(iter.hasNext()) {
				Entry<CellIndex, Double> e = iter.next();
				if(e.getValue() == 0)
					continue;
				values[e.getKey().row].blk.setValue(0, e.getKey().column, e.getValue());
			}
		}
		else {
			double[] darray = thisblock.getDenseArray();
			if (darray != null) {
				long limit = nrows * ncols;
				for(int dd=0; dd<limit; dd++) {
					if(darray[dd] ==0)
						continue;
					int subrowid = dd / ncols;		//we get the row id within the block
					int subcolid = dd - subrowid*ncols;	//darray is in row-major form
					values[subrowid].blk.setValue(0, subcolid, darray[dd]); //TODO: array index is only ints!! can overflow!!
				}
			}
		}
		//recheck sparsity; send out only nonempty subblks!
		for(int r=0; r < nrows; r++) {
			if(values[r].blk.getNonZeros() == 0)		//in alignment with ReBlock's trick, we send only nonempty subblks around!
				continue;
			values[r].blk.setMaxColumn(thisblock.getMaxColumn());	//set maxcol correctly
			values[r].blk.examSparsity();			//refactor based on sparsity
			BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
			key.blkx = blkx;	//the x index is preserved
			if(pp.toReplicate) {
				for(int i = 0; i < pp.numFolds ; i++) {
					long entry = thehashmap.get(blky * rpb + r,i);
					if(entry > 0) {		//the entry is the row index in the test output matrix of fold
						entry--;	//since we started from 1
						key.foldid = 2*i;
					}
					else {
						entry = -1*entry - 1;	//since we started from -1
						key.foldid = 2*i + 1;
					}
					values[r].auxdata = (int) (entry % rpb);	//TODO: could this give wrong results? 
					//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
					key.blky = entry / rpb;	//casting as long itself does flooring
					out.collect(key, values[r]);
				}
			}
			else {	//no replication, output only k test fold entries!
				for(int i = 0; i < pp.numFolds ; i++) {
					long entry = thehashmap.get(blky * rpb + r,i);
					if(entry > 0) {		//the entry is the row index in the test output matrix of fold
						values[r].auxdata = (int) (entry % rpb);	//TODO: could this give wrong results?
						//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
						key.blky = entry / rpb;
						key.foldid = i;
						key.blkx = blkx;	//the x index is preserved
						out.collect(key, values[r]);
					}
				}
			}
		}
		//System.out.println("$$$$ kfold blkhshmpr mapper is done!");		
		*/
		
		
		//the foll is the prev ineff method, whrein we look at all mn cells before sending out kv pairs!		
		/*int nrows = pair.getValue().getNumRows();
		//System.out.println("$$$$$$ in kfold hashmap mapper, torepl is " + pp.toReplicate + " $$$$$$$$$$$");
		//System.out.println("$$$$ blky="+blky+",blkx="+blkx+",rpb="+rpb+",N="+N+",nrows="+nrows+" $$$$$$");
		for(int r=0; r<nrows; r++) {
			BlockHashMapMapOutputValue value = new BlockHashMapMapOutputValue(); 
			value.blk = getSubRowBlock(pair.getValue(), r);
			if(value.blk == null) //in alignment with ReBlock's trick, we send only nonempty subblks around!
				continue;
			BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
			key.blkx = blkx;	//the x index is preserved
			if(pp.toReplicate) {
				for(int i = 0; i < pp.numFolds ; i++) {
					long entry = thehashmap.get(blky * rpb + r)[i];
					if(entry > 0) {		//the entry is the row index in the test output matrix of fold
						entry--;	//since we started from 1
						key.foldid = 2*i;
					}
					else {
						entry = -1*entry - 1;	//since we started from -1
						key.foldid = 2*i + 1;
					}
					value.auxdata = (int) (entry % rpb);	//TODO: could this give wrong results? 
					//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
					key.blky = entry / rpb;	//casting as long itself does flooring
					out.collect(key, value);
				}
			}
			else {	//no replication, output only k test fold entries!
				for(int i = 0; i < pp.numFolds ; i++) {
					long entry = thehashmap.get(blky * rpb + r)[i];
					if(entry > 0) {		//the entry is the row index in the test output matrix of fold
						value.auxdata = (int) (entry % rpb);	//TODO: could this give wrong results?
						//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
						key.blky = entry / rpb;
						key.foldid = i;
						key.blkx = blkx;	//the x index is preserved
						out.collect(key, value);
					}
				}
			}
		}*/
	}
	
	/*@Override
	void execute(WELL1024 currRandom, Pair<MatrixIndexes, MatrixBlock> pair,
			Reporter reporter, OutputCollector out) throws IOException {
		if (pp.toReplicate == false){
			block = pair.getValue() ; 	int partId = currRandom.nextInt(0,pp.numFolds-1) ;
			obj.set(partId) ; out.collect(obj, block) ;
		}else {
			block = pair.getValue() ;	int partId = currRandom.nextInt(0,pp.numFolds-1) ;
			obj.set(2*partId) ;		out.collect(obj, block) ;		
			for(int i = 0 ; i < pp.numFolds; i++) {
				if(i != partId) {	obj.set(2*i + 1) ;		out.collect(obj, block) ;	}
	}}*/
}
//</Arun>