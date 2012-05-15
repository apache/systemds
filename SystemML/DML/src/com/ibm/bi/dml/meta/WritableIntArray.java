package com.ibm.bi.dml.meta;
//<Arun>
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class WritableIntArray implements Writable {
	int length;
	Integer[] array;
	public WritableIntArray() {
		length = 0;
		array = null;
	}
	public void set(int inp, Integer[] arr) {
		length = inp;
		array = arr;
	}
	
	@Override
	public void readFields(DataInput inp) throws IOException {
		length = inp.readInt();
		for(int i=0; i<length; i++)
			array[i] = inp.readInt();
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(length);
		for(int i=0; i<length; i++)
			out.writeInt(array[i]);
	}
}
//</Arun>