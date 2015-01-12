/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public abstract class DataGenMRInstruction extends MRInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public byte input;
	public long rows;
	public long cols;
	public int rowsInBlock;
	public int colsInBlock;
	public String baseDir;
	private DataGenMethod method;
	
	public DataGenMRInstruction(Operator op, DataGenMethod mthd, byte in, byte out, long r, long c, int rpb, int cpb, String dir)
	{
		super(op, out);
		method = mthd;
		input=in;
		rows = r;
		cols = c;
		rowsInBlock = rpb;
		colsInBlock = cpb;
		baseDir = dir;
	}
	
	public DataGenMethod getDataGenMethod() {
		return method;
	}
	
	@Override
	public byte[] getInputIndexes() {
		return new byte[]{input};
	}

	@Override
	public byte[] getAllIndexes() {
		return new byte[]{input, output};
	}

}
