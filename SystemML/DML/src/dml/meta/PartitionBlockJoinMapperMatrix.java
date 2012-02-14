package dml.meta;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import dml.runtime.matrix.io.Converter;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.Pair;
import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.runtime.matrix.mapred.MRJobConfiguration;
import dml.runtime.util.MapReduceTool;

public class PartitionBlockJoinMapperMatrix extends MapReduceBase
implements Mapper<Writable, Writable, LongWritable, BlockJoinMapOutputValue> {
	
	private Converter inputConverter=null;
	PartitionParams pp = new PartitionParams() ;
	int brlen, bclen ;
	MultipleOutputs multipleOutputs ;

	@Override
	public void map(Writable rawKey, Writable rawValue,
			OutputCollector<LongWritable, BlockJoinMapOutputValue> out, Reporter reporter)
	throws IOException {
		inputConverter.setBlockSize(brlen, bclen);
		inputConverter.convert(rawKey, rawValue);
		while(inputConverter.hasNext()) {
			Pair<MatrixIndexes, MatrixBlock> pair=inputConverter.next();
			//bmm.execute(pair, reporter, out) ;
			long blky = pair.getKey().getRowIndex() - 1;
			long blkx = pair.getKey().getColumnIndex() - 1;		//systemml matrixblks start from (1,1)
			int rpb = pp.rows_in_block;		//TODO: assuming this is the general block y dimension
			int cpb = pp.columns_in_block;		//TODO: assuming this is the general block x dimension, usually equal
			
			
			//this is the even-more-eff cell kv pairs version!
			MatrixBlock thisblock = pair.getValue();
			int nrows = thisblock.getNumRows();
			int ncols = thisblock.getNumColumns();
			int nk = (pp.isColumn == true) ? ncols : nrows;
			int nother = (pp.isColumn == false) ? ncols : nrows;
			boolean issparse = thisblock.isInSparseFormat();
			if((issparse == true) && (thisblock.getSparseMap() != null)) {
				Iterator<Entry<CellIndex, Double>> iter = thisblock.getSparseMap().entrySet().iterator();
				while(iter.hasNext()) {
					Entry<CellIndex, Double> e = iter.next();
					if(e.getValue() == 0)
						continue;
					BlockJoinMapOutputValue value = new BlockJoinMapOutputValue();
					if(pp.isColumn == false) {
						value.cellvalue = e.getValue();
						//value.locator = e.getKey().column;
						LongWritable outkey = new LongWritable(blky * rpb + e.getKey().row);	//absol rowid
						value.val2 = 0;	//matx
						value.val1 = blkx * cpb + e.getKey().column;	//absol colid
						out.collect(outkey, value);	//one output key-val pair for each absol rowid
					}
					else {	//subcols
						value.cellvalue = e.getValue();
						//value.locator = e.getKey().row;
						LongWritable outkey = new LongWritable(blkx * cpb + e.getKey().column);	//absol colid
						value.val2 = 0;	//matx
						value.val1 = blky * rpb + e.getKey().row;	//absol rowid
						out.collect(outkey, value);	//one output key-val pair for each absol colid
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
						BlockJoinMapOutputValue value = new BlockJoinMapOutputValue();
						if(pp.isColumn == false) {
							value.cellvalue = darray[dd];
							//value.locator = subcolid;
							LongWritable outkey = new LongWritable(blky * rpb + subrowid);	//absol rowid
							value.val2 = 0;	//matx
							value.val1 = blkx * cpb + subcolid;
							out.collect(outkey, value);	//one output key-val pair for each absol rowid
						}
						else {	//subcols
							value.cellvalue = darray[dd];
							//value.locator = subrowid;
							LongWritable outkey = new LongWritable(blkx * cpb + subcolid);	//absol colid
							value.val2 = 0;	//matx
							value.val1 = blky * rpb + subrowid;
							out.collect(outkey, value);	//one output key-val pair for each absol colid
						}
					}
				}
			}
			
			
			
			//this is the prev eff methods, where i still send subblks around! instead now i send cells around!
			/*//both hm/jr can send out cell kv pairs like ReBlock -> the cost competition is 1/c vs sparsity
			//for very sparse (<0.1%), cell kv pairs might be more efficient since mn/c vs mns => 1/c=0.001 > s!! TODO for now, ignored!
			//we send out subrowblks, but populate them by checking if this blk is dense/sparse! this requires buffering all subblks in mem!
			MatrixBlock thisblock = pair.getValue();
			int nrows = thisblock.getNumRows();
			int ncols = thisblock.getNumColumns();
			int nk = (pp.isColumn == true) ? ncols : nrows;
			int nother = (pp.isColumn == false) ? ncols : nrows;
			//MatrixBlock[] subblocks = new MatrixBlock[nk];
			BlockJoinMapOutputValue[] values = new BlockJoinMapOutputValue[nk];
			for(int r=0; r < nk; r++) {
				values[r] = new BlockJoinMapOutputValue();
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
				values[r].blk.examSparsity();			//refactor based on sparsity
				if(pp.isColumn == false) {
					LongWritable outkey = new LongWritable(blky * rpb + r);	//absol rowid
					values[r].blk.setMaxColumn(thisblock.getMaxColumn());	//set maxcol correctly
					values[r].blk.examSparsity();	//refactor based on sparsity
					values[r].val2 = 0;	//matx
					values[r].val1 = blkx;
					out.collect(outkey, values[r]);	//one output key-val pair for each absol rowid
				}
				else {	//subcols
					LongWritable outkey = new LongWritable(blkx * cpb + r);	//absol colid
					values[r].blk.setMaxRow(thisblock.getMaxRow());	//set maxrow correctly
					values[r].blk.examSparsity();	//refactor based on sparsity
					values[r].val2 = 0;	//matx
					values[r].val1 = blky;
					out.collect(outkey, values[r]);	//one output key-val pair for each absol colid
				}
			}
			//System.out.println("$$$$$ blkjoin mapper is done!");
			*/
			
			//the foll is the prev ineff method, whrein we look at all mn cells before sending out kv pairs!		
			/*//populate subblocks and send them out!
			MatrixBlock matvalue = pair.getValue();
			int nrows = matvalue.getNumRows();
			int ncols = matvalue.getNumColumns();
			int nk = (pp.isColumn == true) ? ncols : nrows;
			for(int r=0; r<nk; r++) {
				if(pp.isColumn == false) {
					LongWritable outkey = new LongWritable(blky * rpb + r);	//absol rowid
					BlockJoinMapOutputValue outval = new BlockJoinMapOutputValue();
					//outval.blkx = blkx;
					MatrixBlock thissubrowblk = new MatrixBlock(1, ncols, true);	//presume sparse
					for(int c=0; c<ncols; c++) {
						thissubrowblk.setValue(0, c, matvalue.getValue(r, c));
					}
					if(thissubrowblk.getNonZeros() == 0) //in alignment with ReBlock's trick, we send only nonempty subblks around!
						continue;
					thissubrowblk.setMaxColumn(matvalue.getMaxColumn());	//set maxcol correctly
					thissubrowblk.examSparsity();	//refactor based on sparsity
					outval.blk = new MatrixBlock();
					outval.blk = thissubrowblk;
					outval.val2 = 0;	//matx
					outval.val1 = blkx;
					out.collect(outkey, outval);	//one output key-val pair for each absol rowid
				}
				else {	//subcols
					LongWritable outkey = new LongWritable(blkx * cpb + r);	//absol colid
					BlockJoinMapOutputValue outval = new BlockJoinMapOutputValue();
					MatrixBlock thissubcolblk = new MatrixBlock(nrows, 1, true);	//presume sparse
					for(int c=0; c<nrows; c++) {
						thissubcolblk.setValue(c, 0, matvalue.getValue(c, r));
					}
					if(thissubcolblk.getNonZeros() == 0) //in alignment with ReBlock's trick, we send only nonempty subblks around!
						continue;
					thissubcolblk.setMaxRow(matvalue.getMaxRow());	//set maxrow correctly
					thissubcolblk.examSparsity();	//refactor based on sparsity
					outval.blk = new MatrixBlock();
					outval.blk = thissubcolblk;
					outval.val2 = 0;	//matx
					outval.val1 = blky;
					out.collect(outkey, outval);	//one output key-val pair for each absol colid
				}
			}*/
			
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
		pp = MRJobConfiguration.getPartitionParams(job) ;
	}
}
