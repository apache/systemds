/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.udf.lib;

import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Matrix.ValueType;
/**
 * Wrapper class for conversions of bit vectors to condensed position vectors.
 * The semantics are equivalent to the following dml snippet:
 *   # bitvector into position vector, e.g., 1011 -> 1034
 *   Bv = seq(1,nrow(B)) * B; 
 *   # gather positions into condensed vector
 *   V = removeEmpty(target=Bv, margin="rows");
 * 
 * Note that the inverse operation would be a scatter that can be implemented 
 * via the following dml snippet:
 *   # position vector into bit vector, e.g., 1034 -> 1011
 *   B = table( V, 1 );
 */
public class GatherWrapper extends PackageFunction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = 1L;
	private final String OUTPUT_FILE = "TMP";

	//return matrix
	private Matrix ret;

	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;	
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{	
		if(pos == 0)
			return ret;
		
		throw new PackageRuntimeException("Invalid function output being requested");
	}

	@Override
	public void execute() 
	{ 
		try 
		{
			//get input and meta information
			Matrix inM = (Matrix) getFunctionInput(0);
			MatrixObject mo = inM.getMatrixObject();
			MatrixBlock mb = mo.acquireRead();
			int len1 = mb.getNumRows();
			int len2 = mb.getNonZeros();
			
			//create condensed position vector
			double[][] outM = new double[len2][1];
			int pos = 0;
			for( int i=0; i<len1; i++ ) {
				double val = mb.quickGetValue(i, 0);
				if( val != 0 )
					outM[pos++][0] = i+1;
			}
			mo.release();
			
			//create and copy output matrix		
			String dir = createOutputFilePathAndName( OUTPUT_FILE );	
			ret = new Matrix( dir, mb.getNonZeros(), 1, ValueType.Double );
			ret.setMatrixDoubleArray(outM);
		} 
		catch (Exception e) 
		{
			throw new PackageRuntimeException("Error executing external order function", e);
		}
	}
}
