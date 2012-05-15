package com.ibm.bi.dml.meta;
//<Arun>
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;


public class BlockHashMapMapOutputValue implements Writable {
	int auxdata;
	//MatrixBlock blk;
	
	//instead of sending out subblks with auxdata indicating posn, i send out the cell value, with another position locator
	//TODO this menas changes in blkhshmapmaputval, btstrpmpr, holdoutmpr, kfoldmpr, and reducer!!
	int locator; //if we send subrowblks, this gives colid within that future blk
	double cellvalue;
	
	public BlockHashMapMapOutputValue() {
		auxdata = 0;
		//blk = new MatrixBlock();
		cellvalue = locator = 0; 
	}
	
	public BlockHashMapMapOutputValue(BlockHashMapMapOutputValue that) {
		auxdata = that.auxdata;
		//blk = new MatrixBlock();
		//blk = that.blk;
		cellvalue = that.cellvalue;
		locator = that.locator;
	}
	@Override
	public void readFields(DataInput inp) throws IOException {
		auxdata = inp.readInt();
		//blk.readFields(inp);
		cellvalue = inp.readDouble();
		locator = inp.readInt();
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(auxdata);
		//blk.write(out);
		out.writeDouble(cellvalue);
		out.writeInt(locator);
	}
	 public void print() {
		 System.out.println(toString());
		 //blk.print();
	 }
		
	 public String toString() {;
	 	return "auxdata: " + auxdata + "\n" 
	 		//+ blk.toString();
	 		+ "locator: " + locator + " cellval: " + cellvalue + "\n";
	 }

}
//</Arun>