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

public class WritableIntArray implements Writable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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