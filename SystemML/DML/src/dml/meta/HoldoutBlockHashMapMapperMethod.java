package dml.meta;
//<Arun>
import java.io.IOException;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.io.MatrixValue.CellIndex;

public class HoldoutBlockHashMapMapperMethod extends BlockHashMapMapperMethod {
	
	public HoldoutBlockHashMapMapperMethod(PartitionParams pp,
			MultipleOutputs multipleOutputs) {
		super(pp, multipleOutputs);
	}

	//Get subrows; send to reducer with map output key <foldid, (blkx, blky)> and  subrowid, matrixblk in value
	//In case of columns, blk y is preserved (not blk x) and subrowid is treated as subcol id!
	@Override
	void execute(Pair<MatrixIndexes, MatrixBlock> pair, Reporter reporter, OutputCollector out)	throws IOException {
		long blky = pair.getKey().getRowIndex() - 1;
		long blkx = pair.getKey().getColumnIndex() - 1;		//systemml matrixblks start from (1,1)
		int rpb = pp.rows_in_block;		//TODO: assuming this is the general block y dimension
		int cpb = pp.columns_in_block;		//TODO: assuming this is the general block x dimension, usually they are equal
		//long N = thehashmap.size();
		long N = thehashmap.length;
		
		///*
		//this is the even-more-eff cell kv pairs version! 1/c vs sparsity decision.
		MatrixBlock thisblock = pair.getValue();
		int nrows = thisblock.getNumRows();
		int ncols = thisblock.getNumColumns();
		int nk = (pp.isColumn == true) ? ncols : nrows;
		int nother = (pp.isColumn == false) ? ncols : nrows;
		BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
		BlockHashMapMapOutputValue value = new BlockHashMapMapOutputValue();
		boolean issparse = thisblock.isInSparseFormat();
		if((issparse == true) && (thisblock.getSparseMap() != null)) {
			Iterator<Entry<CellIndex, Double>> iter = thisblock.getSparseMap().entrySet().iterator();
			while(iter.hasNext()) {
				Entry<CellIndex, Double> e = iter.next();
				if(e.getValue() == 0)
					continue;
				value.cellvalue = e.getValue();
				value.locator = (pp.isColumn == false) ? e.getKey().column : e.getKey().row;
				int numtimes = (pp.toReplicate == false) ? 1 : pp.numIterations;
				for(int i = 0; i < numtimes ; i++) {
					//long entry = (pp.isColumn == false) ? thehashmap.get( (blky * rpb + e.getKey().row))[i] :  
					//												thehashmap.get( (blkx * cpb + e.getKey().column))[i];
					//long entry = (pp.isColumn == false) ? thehashmap.get((blky * rpb + e.getKey().row),i) :  
					//	thehashmap.get((blkx * cpb + e.getKey().column),i);
					long entry = (pp.isColumn == false) ? thehashmap.get((blky * rpb + e.getKey().row),i).get(0) :  
						(thehashmap.get((blkx * cpb + e.getKey().column),i)).get(0);	//retvals is a vector, but with single entry
					if(entry > 0) {		//the entry is the row index in the test output matrix of fold; ignore for el
						if(pp.isEL == true)
							continue;
						entry--;	//since we started from 1
						key.foldid = 2*i;
					}
					else {	//train set
						entry = -1*entry - 1;	//since we started from -1
						key.foldid = (pp.isEL == true) ? i : 2*i + 1;
					}
					if(pp.isColumn == false) {
						value.auxdata = (int) (entry % rpb);
						//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
						key.blky = entry / rpb;		//by casting as long, we are effectively taking the floor value
						key.blkx = blkx;	//the x index is preserved
					}
					else {
						value.auxdata = (int) (entry % cpb);
						//key.blkx = (long) Math.floor(entry / cpb);		//TODO: this may give wrong result!!
						key.blkx = entry / cpb;
						key.blky = blky;	//the y index is preserved
					}
					out.collect(key, value);
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
					value.locator = (pp.isColumn == false) ? subcolid : subrowid;
					int numtimes = (pp.toReplicate == false) ? 1 : pp.numIterations;
					for(int i = 0; i < numtimes ; i++) {
						long entry = (pp.isColumn == false) ? thehashmap.get((blky * rpb + subrowid),i).get(0) :  
																		thehashmap.get((blkx * cpb + subcolid),i).get(0);
						if(entry > 0) {		//the entry is the row index in the test output matrix of fold; ignore for el
							if(pp.isEL == true)
								continue;
							entry--;	//since we started from 1
							key.foldid = 2*i;
						}
						else {	//train set
							entry = -1*entry - 1;	//since we started from -1
							key.foldid = (pp.isEL == true) ? i : 2*i + 1;
						}
						if(pp.isColumn == false) {
							value.auxdata = (int) (entry % rpb);
							//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
							key.blky = entry / rpb;		//by casting as long, we are effectively taking the floor value
							key.blkx = blkx;	//the x index is preserved
						}
						else {
							value.auxdata = (int) (entry % cpb);
							//key.blkx = (long) Math.floor(entry / cpb);		//TODO: this may give wrong result!!
							key.blkx = entry / cpb;
							key.blky = blky;	//the y index is preserved
						}
						out.collect(key, value);
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
		int nk = (pp.isColumn == true) ? ncols : nrows;
		int nother = (pp.isColumn == false) ? ncols : nrows;
		//MatrixBlock[] subblocks = new MatrixBlock[nk];
		BlockHashMapMapOutputValue[] values = new BlockHashMapMapOutputValue[nk]; 
		for(int r=0; r < nk; r++) {
			values[r] = new BlockHashMapMapOutputValue();
			values[r].blk = (pp.isColumn == false) ? new MatrixBlock(1, ncols, true) : new MatrixBlock(nrows, 1, true);	//presume sparse!
		}
		boolean issparse = thisblock.isInSparseFormat();
		if((issparse == true) && (thisblock.getSparseMap() != null)) {
			Iterator<Entry<CellIndex, Double>> iter = thisblock.getSparseMap().entrySet().iterator();
			while(iter.hasNext()) {
				Entry<CellIndex, Double> e = iter.next();
				if(e.getValue() == 0)
					continue;
				if(pp.isColumn == false)
					values[e.getKey().row].blk.setValue(0, e.getKey().column, e.getValue());
				else
					values[e.getKey().column].blk.setValue(e.getKey().row, 0, e.getValue());
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
					if(pp.isColumn == false)
						values[subrowid].blk.setValue(0, subcolid, darray[dd]); //TODO: array index is only ints!! can overflow!!
					else
						values[subcolid].blk.setValue(subrowid, 0, darray[dd]);
				}
			}
		}
		//recheck sparsity; send out only nonempty subblks!
		for(int r=0; r < nk; r++) {
			if(values[r].blk.getNonZeros() == 0)		//in alignment with ReBlock's trick, we send only nonempty subblks around!
				continue;
			if(pp.isColumn == true)
				values[r].blk.setMaxRow(thisblock.getMaxRow());	//set maxrow correctly
			else
				values[r].blk.setMaxColumn(thisblock.getMaxColumn());	//set maxcol correctly
			values[r].blk.examSparsity();			//refactor based on sparsity
			BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
			int numtimes = (pp.toReplicate == false) ? 1 : pp.numIterations;
			for(int i = 0; i < numtimes ; i++) {
				long entry = (pp.isColumn == false) ? thehashmap.get((blky * rpb + r),i) :  thehashmap.get((blkx * cpb + r),i);
				if(entry > 0) {		//the entry is the row index in the test output matrix of fold; ignore for el
					if(pp.isEL == true)
						continue;
					entry--;	//since we started from 1
					key.foldid = 2*i;
				}
				else {	//train set
					entry = -1*entry - 1;	//since we started from -1
					key.foldid = (pp.isEL == true) ? i : 2*i + 1;
				}
				if(pp.isColumn == false) {
					values[r].auxdata = (int) (entry % rpb);
					//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
					key.blky = entry / rpb;		//by casting as long, we are effectively taking the floor value
					key.blkx = blkx;	//the x index is preserved
				}
				else {
					values[r].auxdata = (int) (entry % cpb);
					//key.blkx = (long) Math.floor(entry / cpb);		//TODO: this may give wrong result!!
					key.blkx = entry / cpb;
					key.blky = blky;	//the y index is preserved
				}
				out.collect(key, values[r]);
			}
		}
		//System.out.println("$$$$$ holfout blkhshmp mapper is done!");
		*/
		
		//the foll is the prev ineff method, whrein we look at all mn cells before sending out kv pairs!		
		/*int nrows = pair.getValue().getNumRows();
		int ncols = pair.getValue().getNumColumns();
		int nk = (pp.isColumn == true) ? ncols : nrows;
		for(int r=0; r<nk; r++) {
			BlockHashMapMapOutputValue value = new BlockHashMapMapOutputValue(); 
			value.blk = (pp.isColumn == true) ? getSubColBlock(pair.getValue(), r) : getSubRowBlock(pair.getValue(), r);
			if(value.blk == null) //in alignment with ReBlock's trick, we send only nonempty subblks around!
				continue;
			BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
			int numtimes = (pp.toReplicate == false) ? 1 : pp.numIterations;
			for(int i = 0; i < numtimes ; i++) {
				long entry = (pp.isColumn == false) ? thehashmap.get( (blky * rpb + r))[i] :  thehashmap.get( (blkx * cpb + r))[i];
				if(entry > 0) {		//the entry is the row index in the test output matrix of fold; ignore for el
					if(pp.isEL == true)
						continue;
					entry--;	//since we started from 1
					key.foldid = 2*i;
				}
				else {	//train set
					entry = -1*entry - 1;	//since we started from -1
					key.foldid = (pp.isEL == true) ? i : 2*i + 1;
				}
				if(pp.isColumn == false) {
					value.auxdata = (int) (entry % rpb);
					//key.blky = (long) Math.floor(entry / rpb);		//TODO: this may give wrong result!!
					key.blky = entry / rpb;		//by casting as long, we are effectively taking the floor value
					key.blkx = blkx;	//the x index is preserved
				}
				else {
					value.auxdata = (int) (entry % cpb);
					//key.blkx = (long) Math.floor(entry / cpb);		//TODO: this may give wrong result!!
					key.blkx = entry / cpb;
					key.blky = blky;	//the y index is preserved
				}
				out.collect(key, value);
			}
		}*/
		
	}
}
//</Arun>