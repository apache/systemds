/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

public enum RuntimeDataFormat 
{
	Invalid(true), 
	TextCell(true), 
	BinaryCell(true), 
	BinaryBlock(true), 
	SortInput(true), 
	SortOutput(true), 
	WeightedPair(true), 
	MatrixMarket(false);
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean nativeFormat = false;
	private RuntimeDataFormat(boolean nf) {
		nativeFormat = nf;
	}
	
	public static RuntimeDataFormat parseFormat(String fmt) {
		if(fmt.equalsIgnoreCase("textcell")) 
			return TextCell;
		else if(fmt.equalsIgnoreCase("binarycell"))
			return BinaryCell;
		else if(fmt.equalsIgnoreCase("binaryblock"))
			return BinaryBlock;
		else if(fmt.equalsIgnoreCase("sort_input"))
			return SortInput;
		else if(fmt.equalsIgnoreCase("sort_output"))
			return SortOutput;
		else if(fmt.equalsIgnoreCase("weightedpair"))
			return WeightedPair;
		else if(fmt.equalsIgnoreCase("matrixmarket"))
			return MatrixMarket;
		else 
			return Invalid;
	}

	public boolean isNative() {
		return nativeFormat;
	}

}
