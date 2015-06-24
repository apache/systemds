/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import org.apache.spark.api.java.function.Function2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;

/**
 * 
 */
public class ResultMergeRemoteSparkWoCompare extends ResultMerge implements Function2<MatrixBlock,MatrixBlock,MatrixBlock> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final long serialVersionUID = 9151776096399086821L;
	
	@Override
	public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1) 
		throws Exception 
	{
		//copy first argument to output
		MatrixBlock out = new MatrixBlock(arg0);
		
		//merge second argument into output
		mergeWithoutComp(out, arg1, false);
		
		return out;
	}


	@Override
	public MatrixObject executeSerialMerge() 
			throws DMLRuntimeException 
	{
		throw new DMLRuntimeException("Unsupported operation.");
	}

	@Override
	public MatrixObject executeParallelMerge(int par)
			throws DMLRuntimeException 
	{
		throw new DMLRuntimeException("Unsupported operation.");
	}
}
