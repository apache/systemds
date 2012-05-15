package com.ibm.bi.dml.runtime.matrix;

public class MatrixDimensionsMetaData extends MetaData {
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
	
	public String toString() {
		return "[rows = " + matchar.numRows + 
			   ", cols = " + matchar.numColumns + 
			   ", rpb = " + matchar.numRowsPerBlock + 
			   ", cpb = " + matchar.numColumnsPerBlock + "]"; 
	}
}
