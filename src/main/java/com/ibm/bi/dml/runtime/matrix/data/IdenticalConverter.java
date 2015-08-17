/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import org.apache.hadoop.io.Writable;

public class IdenticalConverter implements Converter<Writable, Writable, Writable, Writable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Pair<Writable, Writable> pair=new Pair<Writable, Writable>();
	private boolean hasValue=false;
	
	@Override
	public void convert(Writable k1, Writable v1) {
		pair.set(k1, v1);
		hasValue=true;
	}

	public boolean hasNext() {
		return hasValue;
	}


	public Pair<Writable, Writable> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}

	@Override
	public void setBlockSize(int rl, int cl) {
	}
}
