package com.ibm.bi.dml.runtime.matrix.io;

import java.util.HashMap;

public class MatrixBlock extends MatrixBlockDSM
{
	public MatrixBlock() {
		super();
	}
	
	public MatrixBlock(int rlen, int clen, boolean sparse) {
		super(rlen, clen, sparse);
	}

	public MatrixBlock(int rlen, int clen, boolean sparse, int estnnzs){
		super(rlen, clen, sparse, estnnzs);
	}
	
	public MatrixBlock(HashMap<CellIndex, Double> map) {
		super(map);
	}
	
	
	public static MatrixBlock randOperations(int rows, int cols, double sparsity, double min, double max, String pdf, long seed)
	{
		MatrixBlock m = new MatrixBlock();
		
		if ( pdf.equalsIgnoreCase("normal") ) {
			m.getNormalRandomSparseMatrix(rows, cols, sparsity, seed);
		}
		else {
			m.getRandomSparseMatrix(rows, cols, sparsity, min, max, seed);
		}
		return m;
	}
}
