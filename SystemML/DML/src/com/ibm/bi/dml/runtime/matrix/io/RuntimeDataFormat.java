package com.ibm.bi.dml.runtime.matrix.io;

public enum RuntimeDataFormat {
	Invalid(true), 
	TextCell(true), 
	BinaryCell(true), 
	BinaryBlock(true), 
	SortInput(true), 
	SortOutput(true), 
	WeightedPair(true), 
	MatrixMarket(false);
	
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
