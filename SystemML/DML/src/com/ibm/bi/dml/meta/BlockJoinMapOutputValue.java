package com.ibm.bi.dml.meta;
//<Arun>
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;


public class BlockJoinMapOutputValue implements Writable {
	public long val1;	//used as blk.x/y for matrix read; as future row/col id in bootstrap / holdout/kfold idtable read (repl coded)
	public int val2;	//used as foldnum for btstrp tupleswp (+ve +1) / holdout/kfold idtable read (-ve -1) / 0 for matx
	//public MatrixBlock blk;	//blk val for matrix file read
	//instead of sending subrowcolblk, i send cell-value with colindex within full row! (rows) / rowindex within full col! (cols)! - val1 is used!
	//then, i can reconstruct cell at reducer! - this shld reduce comptn time at mapper/reducer! thus now, we have 1xnc blks output
	//TODO changes in cvpgmblk, elpgmblk,blkjoinmapopvalue, partnblkjoinreducer, partnblkjoinmappermatrix)
	public double cellvalue;
	
	public BlockJoinMapOutputValue() {
		val1 = -1; val2 = 0;
		//blk = null;
		cellvalue = 0;
	}
	public BlockJoinMapOutputValue(BlockJoinMapOutputValue that) {
		if(that.val2 == 0) {	//matx
			//blk = new MatrixBlock();
			//blk = that.blk;
			cellvalue = that.cellvalue;
		}
		val1 = that.val1;
		val2 = that.val2;
	}
	
	@Override
	public void readFields(DataInput inp) throws IOException {
		val1 = inp.readLong();
		val2 = inp.readInt();
		if(val2 == 0) {	//only for matx
			//blk = new MatrixBlock();
			//blk.readFields(inp);
			cellvalue = inp.readDouble();
		}
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(val1);
		out.writeInt(val2);
		if(val2 == 0) {	//only for matx
			//blk.write(out);
			out.writeDouble(cellvalue);
		}
	}
	 public void print() {
		 System.out.println(toString());
	 }
		
	 public String toString() {;
	 	String tmp = "val1: " + val1 + ", val2: " + val2 + "\n";
	 	//if(blk != null)
	 	//	tmp += blk.toString() + "\n";
	 	tmp += "cellvalue: " + cellvalue + "\n";
	 	return tmp;
	 }

	/*long blkx;			//blk.x for matrix file read
	MatrixBlock blk;	//blk val for matrix file read
	long futrowid;		//from bootstrap idtable tuple swap
	long foldnum;		//from bootstrap idtable tuple swap
	WritableLongArray futrowids;	//from holdout/kfold idtable read
	
	public BlockJoinMapOutputValue() {
		blkx = futrowid = foldnum = -1;
		blk = new MatrixBlock();
		futrowids = new WritableLongArray();
	}
	public BlockJoinMapOutputValue(BlockJoinMapOutputValue that) {
		blkx = that.blkx;
		futrowid = that.futrowid;
		foldnum = that.foldnum;
		blk = new MatrixBlock();
		blk = that.blk;
		futrowids = new WritableLongArray(that.futrowids);
	}
	
	@Override
	public void readFields(DataInput inp) throws IOException {
		blkx = inp.readLong();
		blk.readFields(inp);
		futrowid = inp.readLong();
		foldnum = inp.readLong();
		futrowids.readFields(inp);
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(blkx);
		blk.write(out);
		out.writeLong(futrowid);
		out.writeLong(foldnum);
		futrowids.write(out);
	}
	 public void print() {
		 System.out.println("longdata: " + blkx);
		 blk.print();
		 System.out.println("futrowid: " + futrowid);
		 System.out.println("foldnum: " + foldnum);
		 futrowids.print();
	 }
		
	 public String toString() {;
	 	return "longdata: " + blkx + "\n" + blk.toString() + "\n" + "futrowid: " + futrowid + "\n" 
	 				+ "foldnum : " + foldnum + "\n" + futrowids.toString(); 
	 }*/
}
//</Arun>