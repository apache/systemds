/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;

public class MatrixFormatMetaData extends MatrixDimensionsMetaData 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private InputInfo iinfo;
	private OutputInfo oinfo;
	
	public MatrixFormatMetaData(MatrixCharacteristics mc, OutputInfo oinfo_, InputInfo iinfo_ ) {
		super(mc);
		oinfo = oinfo_;
		iinfo = iinfo_;
	}
	
	public InputInfo getInputInfo() {
		return iinfo;
	}
	
	public OutputInfo getOutputInfo() {
		return oinfo;
	}
	
	@Override
	public Object clone()
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(matchar);
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, oinfo, iinfo);
		
		return meta;
	}
}
