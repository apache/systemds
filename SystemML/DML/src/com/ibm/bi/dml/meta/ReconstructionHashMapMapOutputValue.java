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



public class ReconstructionHashMapMapOutputValue implements Writable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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