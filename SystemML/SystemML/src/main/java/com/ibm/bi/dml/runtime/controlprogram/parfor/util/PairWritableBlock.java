/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.hadoop.io.Writable;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;

/**
 * Custom writable for a pair of matrix indexes and matrix block
 * as required for binaryblock in remote data partitioning.
 * 
 */
public class PairWritableBlock implements Writable, Serializable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = -6022511967446089164L;

	public MatrixIndexes indexes;
	public MatrixBlock block;
	
	@Override
	public void readFields(DataInput in) throws IOException 
	{
		indexes = new MatrixIndexes();
		indexes.readFields(in);
		
		block = new MatrixBlock();
		block.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException 
	{
		indexes.write(out);
		block.write(out);
	}
}
