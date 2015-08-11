/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class IsBlockInRange implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Boolean> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 5849687296021280540L;
	
	private long _rl; long _ru; long _cl; long _cu;
	private int _brlen; int _bclen;
	
	public IsBlockInRange(long rl, long ru, long cl, long cu, int brlen, int bclen) {
		_rl = rl;
		_ru = ru;
		_cl = cl;
		_cu = cu;
		_brlen = brlen;
		_bclen = bclen;
	}

	@Override
	public Boolean call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
		throws Exception 
	{
		return UtilFunctions.isInBlockRange(kv._1(), _brlen, _bclen, _rl, _ru, _cl, _cu);
	}
}
