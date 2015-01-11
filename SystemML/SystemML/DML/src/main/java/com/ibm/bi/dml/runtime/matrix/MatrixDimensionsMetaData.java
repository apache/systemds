/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

public class MatrixDimensionsMetaData extends MetaData 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	protected MatrixCharacteristics matchar;
	
	public MatrixDimensionsMetaData() {
		matchar = null;
	}
	
	public MatrixDimensionsMetaData(MatrixCharacteristics mc) {
		matchar = mc;
	}
	
	public MatrixCharacteristics getMatrixCharacteristics() {
		return matchar;
	}
	
	public void setMatrixCharacteristics(MatrixCharacteristics mc) {
		matchar = mc;
	}
	
	@Override
	public boolean equals (Object anObject)
	{
		if (anObject instanceof MatrixDimensionsMetaData)
		{
			MatrixDimensionsMetaData md = (MatrixDimensionsMetaData) anObject;
			return (matchar.equals (md.matchar));
		}
		else
			return false;
	}
	
	@Override
	public int hashCode()
	{
		//use identity hash code
		return super.hashCode();
	}

	@Override
	public String toString() {
		return "[rows = " + matchar.numRows + 
			   ", cols = " + matchar.numColumns + 
			   ", rpb = " + matchar.numRowsPerBlock + 
			   ", cpb = " + matchar.numColumnsPerBlock + "]"; 
	}
}
