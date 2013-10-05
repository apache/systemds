/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;
//<Arun>
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class WritableLongArray implements Writable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	int length;
	Long[] array;
	public WritableLongArray() {
		length = 0;
		array = null;	//TODO; need to change this!
	}
	public WritableLongArray(int i) {
		length = i;
		array = new Long[length];
	}
	public WritableLongArray(WritableLongArray that) {
		length = that.length;
		array = new Long[length];
		array = that.array;
	}
	public WritableLongArray(int inp, Long[] arr) {
		length = inp;
		array = new Long[length];
		array = arr;
	}
	public void set(int inp, Long[] arr) {
		length = inp;
		array = new Long[length];
		array = arr;
	}
	@Override
	public void readFields(DataInput inp) throws IOException {
		length = inp.readInt();
		if(length > 0) {
			array = new Long[length];
			for(int i=0; i<length; i++) {
				array[i] = inp.readLong();
			}
		}
		else {
			array = null;
		}
	}
	@Override
	public void write(DataOutput out) throws IOException {
		out.writeInt(length);
		if(length > 0) {
		for(int i=0; i<length; i++)
			out.writeLong(array[i]);
		}
	}
	 public void print() {
		 String getstr = toString();
		 System.out.println(getstr);
	 }
		
	 public String toString() {;
	 	String ret = "length="+ length;
	 	if(length > 0) {
	 		ret += " & array=" + array[0];
			for(int i=1; i<length; i++)
				ret += "," + array[i];
		}
	 	return ret;
	 }

}
//</Arun>