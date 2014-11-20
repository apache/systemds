/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;
//<Arun>
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.ibm.bi.dml.runtime.matrix.data.IJV;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.Pair;
import com.ibm.bi.dml.runtime.matrix.data.SparseRowsIterator;


public class BootstrapBlockHashMapMapperMethod extends BlockHashMapMapperMethod 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public BootstrapBlockHashMapMapperMethod(PartitionParams pp,
			MultipleOutputs multipleOutputs) {
		super(pp, multipleOutputs);
	}

	@Override
	void execute(Pair<MatrixIndexes, MatrixBlock> pair, Reporter reporter, OutputCollector out)	throws IOException {
		long blky = pair.getKey().getRowIndex() - 1;
		long blkx = pair.getKey().getColumnIndex() - 1;	//systemml matrixblks start from (1,1)
		int rpb = pp.rows_in_block;		//TODO: assuming this is the general block y dimension
		//long N = thehashmap.size();		//the hashmap size will be num rows of fut train matrices 
		long N = thehashmap.length;
		
		
		///*
		//this is the new more eff method of sending out cells as kv pairs, due to 1/c vs sparsity, as done by ReBlock
		//first go thro hashmap and get keys (futrowids) where any of this blk's rowids (blky*rpb to blky*rpb + rpb-1) occur as values 
		//ie we do an invert and select on the hashmap; we create one hashmap per fold; the key is the prevrowid, the value is list of futrowids
		int numtimes = (pp.toReplicate == true) ? pp.numIterations : 1;
		
		//the foll forloops caused runtime to become quadratic! so, i introduced vectorofarraybag hashmap, which is more efficieint
		//HashMap<Long, Vector<Vector<Long>>> invselmaps = new HashMap<Long, Vector<Vector<Long>>>();
		//for(long r=blky*rpb; r < blky*rpb+rpb; r++) {
		//	invselmaps.put(r, new Vector<Vector<Long>>());
		//	for(int f=0; f<numtimes; f++) {
		//		invselmaps.get(r).add(new Vector<Long>());
		//	}
		//}
		//for(long yi=0; yi<thehashmap.length; yi++){	//yi is the futrowid
		//	for(int xi=0; xi<thehashmap.width; xi++){	//xi is the foldnum
		//		long prevrowid = thehashmap.get(yi,xi);
		//		if(prevrowid >= blky*rpb && prevrowid < blky*rpb+rpb)
		//			invselmaps.get(prevrowid).get(xi).add(yi);
		//	}	
		//}
	
		//now, we go thro the cells in this blk, and use the hashmap to output kv pairs!
		MatrixBlock thisblock = pair.getValue();
		int nrows = thisblock.getNumRows();
		int ncols = thisblock.getNumColumns();
		BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
		key.blkx = blkx;					//the futblk x index is preserved
		boolean issparse = thisblock.isInSparseFormat();
		if( issparse ) {
			SparseRowsIterator citer = thisblock.getSparseRowsIterator();
			while(citer.hasNext()) {
				IJV e = citer.next();
				BlockHashMapMapOutputValue value = new BlockHashMapMapOutputValue(); 
				value.cellvalue = e.v;
				value.locator = e.j;	//the subcolid within fut blk will be the same!
				long absolprevrowid = blky*rpb+e.i;
				for(int f=0; f<numtimes; f++) { //iterate thro all fut rowids in a fold for this prevrowid, and repeat for all folds
					//Iterator fiter = invselmaps.get(absolprevrowid).get(f).iterator();					
					Iterator fiter = thehashmap.get(absolprevrowid, f).iterator();	//this gives us a vector of longs (futrowdids) 
					while(fiter.hasNext()) {
						long afutrowid = (Long)fiter.next();
						key.foldid = f;
						key.blky = afutrowid / rpb;
						value.auxdata = (int) (afutrowid % rpb); //subrowid within future blk; casting can cause overflow! TODO
						out.collect(key, value);
					}
				}
			}
		}
		else {
			double[] darray = thisblock.getDenseArray();
			if (darray != null) {
				long limit = nrows * ncols;
				for(int dd=0; dd<limit; dd++) {
					if(darray[dd] ==0)
						continue;
					BlockHashMapMapOutputValue value = new BlockHashMapMapOutputValue(); 
					int subrowid = dd / ncols;		//we get the row id within the block
					int subcolid = dd - subrowid*ncols;	//darray is in row-major form
					value.cellvalue = darray[dd];
					value.locator = subcolid;
					long absolprevrowid = blky*rpb+subrowid;
					for(int f=0; f<numtimes; f++) { //iterate thro all fut rowids in all folds for this prevrowid
						//Iterator fiter = invselmaps.get(absolprevrowid).get(f).iterator();					
						Iterator fiter = thehashmap.get(absolprevrowid, f).iterator();	//this gives us a vector of longs (futrowdids) 
						while(fiter.hasNext()) {
							long afutrowid = (Long)fiter.next();
							key.foldid = f;
							key.blky = afutrowid / rpb;
							value.auxdata = (int) (afutrowid % rpb); //subrowid within future blk; casting can cause overflow! TODO
							out.collect(key, value);
						}
					}					
				}
			}
		}
		//*/		
		
		/*
		//the foll is the prev eff sublk based method; instead i send out cell kv pairs now, using ReBlock's trick (1/c vs sparsity)
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
		//recheck sparsity; if empty, set blk to null! later, ignore the kv pairs where blk is null!
		for(int r=0; r < nrows; r++) {
			if(values[r].blk.getNonZeros() == 0) {		//in alignment with ReBlock's trick, we send only nonempty subblks around!
				values[r] = null;
				continue;
			}
			values[r].blk.setMaxColumn(thisblock.getMaxColumn());	//set maxcol correctly
			values[r].blk.examSparsity();			//refactor based on sparsity
		}
		//now go thro hashmap and get keys (futrowids) where thisrowid occurs as value in this fold! and send out kv pairs!
		//Iterator<Entry<Long, Long[]>> iter = thehashmap.entrySet().iterator();
		//while(iter.hasNext()) {
		//	Entry<Long, Long[]> e = iter.next();
		//	long futrowid = e.getKey();
		for(long yi=0; yi<thehashmap.length; yi++){	//yi is the futrowid
			BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
			key.blkx = blkx;					//the x index is preserved
			//int numtimes = (pp.toReplicate == false) ? 1 : pp.numIterations;	//this shld match e's arrlen
			//for(int i = 0; i < numtimes ; i++) {	//only trainfolds for bootstrap
			for(int xi=0; xi<thehashmap.width; xi++){	//xi is the foldnum
				//long prevrowid = e.getValue()[i];
				long prevrowid = thehashmap.get(yi,xi);
				if((prevrowid / rpb) != blky) //if the prevrowid is not within thisblock, ignore!
					continue;
				int prevsubrowid = (int) (prevrowid % rpb);	//the cast shld not cause probs
		//System.out.println("$$$$$ btstrpHMmpr has futrowid"+futrowid+" i"+i+" rpb"+rpb+" entry"+e.getValue()[i]+" prevsubrowid"+prevsubrowid+
		//		"while values length is "+values.length);
				if(values[prevsubrowid] == null)	//ignore kv pair if subblk is empty!
					continue;
				key.foldid = xi;//i;
				key.blky = yi / rpb;//futrowid / rpb;
				values[prevsubrowid].auxdata = (int) (yi % rpb);//(futrowid % rpb);	//subrowid within future blk
				out.collect(key, values[prevsubrowid]);
			}
		}

		//System.out.println("$$$$$ btstrp blkHmmpr mapper is done!");
		*/
		
		
		//the foll is the prev ineff method, whrein we look at all mn cells before sending out kv pairs!
		/*int nrows = pair.getValue().getNumRows();
		//System.out.println("$$$$$$$$ in btstrp blkhshmpmapper blky:"+blky+",blx:"+blkx+",nrows:"+nrows+",N:"+N+" $$$$$$$$$$$");
		for(int r=0; r < nrows; r++) {
			long thisrowid = blky * rpb + r;	//(blky - 1) * rpb + r;	//->blkx/y start from 0!
			BlockHashMapMapOutputValue value = new BlockHashMapMapOutputValue(); 
			value.blk = getSubRowBlock(pair.getValue(), r);
			if(value.blk == null) //in alignment with ReBlock's trick, we send only nonempty subblks around!
				continue;
			BlockHashMapMapOutputKey key = new BlockHashMapMapOutputKey();
			key.blkx = blkx;	//the x index is preserved
			int numtimes = (pp.toReplicate == false) ? 1 : pp.numIterations;
			for(int i = 0; i < numtimes ; i++) {	//only trainfolds for bootstrap
				key.foldid = i;
				//now go thro hashmap and get keys (futrowids) where thisrowid occurs as value in this fold!
				for(long k=0; k<N; k++) {
					if(thehashmap.get(k)[i] == thisrowid) {	//located one, send it out!
						value.auxdata = (int) (k % rpb);	//subrowid within future blk
						key.blky = k / rpb;					//casting as long itself floors it
						out.collect(key, value);
					}
				}
			}
		}*/
	}
}
//</Arun>