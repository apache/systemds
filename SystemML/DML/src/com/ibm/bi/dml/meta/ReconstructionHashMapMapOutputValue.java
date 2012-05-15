package com.ibm.bi.dml.meta;
//<Arun>
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;


public class ReconstructionHashMapMapOutputValue implements Writable {
	int subrowid;	//assuming W is a col matrix for reconstruction
	double entry;	//the actual entry value
	public ReconstructionHashMapMapOutputValue() {
		entry = subrowid = 0;
	}
	public ReconstructionHashMapMapOutputValue(int s, double e) {
		subrowid = s;
		entry = e;
	}
	public ReconstructionHashMapMapOutputValue(ReconstructionHashMapMapOutputValue that) {
		subrowid = that.subrowid;
		entry = that.entry;
	}
	@Override
	public void readFields(DataInput inp) throws IOException {
		subrowid = inp.readInt();
		entry = inp.readDouble();
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(subrowid);
		out.writeDouble(entry);
	}
	 public void print() {
		 System.out.println(toString());
	 }
		
	 public String toString() {;
	 	return "subrowid: " + subrowid + ", entry: " + entry;
	 }

}
//</Arun>