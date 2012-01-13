package dml.meta;
//<Arun>
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

import dml.runtime.matrix.io.MatrixBlock;

public class ReconstructionJoinMapOutputValue implements Writable {
	long rowid;	//assuming W is a col matrix for reconstruction
	double entry;	//the actual entry value
	public ReconstructionJoinMapOutputValue() {
		entry = rowid = 0;
	}
	public ReconstructionJoinMapOutputValue(long s, double e) {
		rowid = s;
		entry = e;
	}
	public ReconstructionJoinMapOutputValue(ReconstructionJoinMapOutputValue that) {
		rowid = that.rowid;
		entry = that.entry;
	}
	@Override
	public void readFields(DataInput inp) throws IOException {
		rowid = inp.readLong();
		entry = inp.readDouble();
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeLong(rowid);
		out.writeDouble(entry);
	}
	 public void print() {
		 System.out.println(toString());
	 }
		
	 public String toString() {;
	 	return "truncrowid: " + rowid + ", entry: " + entry;
	 }

}
//</Arun>