/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;

public class MergeBlocksFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -8881019027250258850L;

	@Override
	public MatrixBlock call(MatrixBlock b1, MatrixBlock b2) 
		throws Exception 
	{
		// sanity check input dimensions
		if (b1.getNumRows() != b2.getNumRows() || b1.getNumColumns() != b2.getNumColumns()) {
			throw new DMLRuntimeException("Mismatched block sizes: "
					+ b1.getNumRows() + " " + b1.getNumColumns() + " "
					+ b2.getNumRows() + " " + b2.getNumColumns());
		}

		// execute merge (never pass by reference)
		MatrixBlock ret = new MatrixBlock(b1);
		ret.merge(b2, false);

		// sanity check output number of non-zeros
		if (ret.getNonZeros() != b1.getNonZeros() + b2.getNonZeros()) {
			throw new DMLRuntimeException("Number of non-zeros does not match: "
					+ ret.getNonZeros() + " != " + b1.getNonZeros() + " + " + b2.getNonZeros());
		}

		return ret;
	}

}