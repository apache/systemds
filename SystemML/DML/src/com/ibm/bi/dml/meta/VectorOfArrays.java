/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.util.Vector;

public class VectorOfArrays 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public Vector<long[]> thevec;	//each long[] is of length 1<<30 
	public long length;		//num rows/cols
	int width;		//num folds/iterns
	public VectorOfArrays() {
		thevec = null;
		length = width = 0;
	}
	public VectorOfArrays(long l, int w) {
		length = l;
		width = w;
		long numarrays = (length * width) / (1 << 30);	//typically, very few arrays, since each is 1bln!
		thevec = new Vector<long[]>();
		for(long i = 0; i < numarrays-1; i++) {
			thevec.add(new long[(1<<30)]);	//the system will possibly run of out memory anyways (each array is 4GB!)
		}
		int leftover = (int) ((length * width) - (numarrays * (1 << 30))); //cast wont cause problems
		thevec.add(new long[leftover]);	
	}
	public VectorOfArrays(VectorOfArrays that) {
		length = that.length;
		width = that.width;
		thevec = new VectorOfArrays(length, width).thevec;
		thevec = that.thevec;
	}

	public void set(long yindex, int xindex, long value) {
		long absolindex = yindex * width + xindex;	//since we store in row major order
		int arraynum = (int)(absolindex / (1 << 30));	//the cast wont cause overflows
		int leftover = (int) (absolindex - arraynum * (1 << 30));
		thevec.get(arraynum)[leftover] = value;
	}

	public long get(long yindex, int xindex) {
		long absolindex = yindex * width + xindex;	//since we store in row major order
		int arraynum = (int)(absolindex / (1 << 30));	//the cast wont cause overflows
		int leftover = (int) (absolindex - arraynum * (1 << 30));
		return thevec.get(arraynum)[leftover];
	}
}
