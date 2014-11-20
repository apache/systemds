/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.meta;

import java.io.IOException;

import org.apache.hadoop.mapred.OutputCollector;

import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.Pair;


public class RowMapperMethod extends MapperMethod 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public RowMapperMethod(PartitionParams pp) {
		super(pp) ;
	}

	@Override
	void execute(Pair<MatrixIndexes, MatrixValue> pair, OutputCollector out)
			throws IOException {
		MatrixCell value = (MatrixCell) pair.getValue() ;
		pols.set(pair.getKey().getRowIndex()) ;
		partialBuffer.set(1, (int) pair.getKey().getColumnIndex(), value.getValue()) ;
		out.collect(pols, partialBuffer) ;
	}
}
