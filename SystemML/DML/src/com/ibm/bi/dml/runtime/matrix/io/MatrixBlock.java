package com.ibm.bi.dml.runtime.matrix.io;

import java.util.HashMap;

import org.apache.commons.math.random.Well1024a;

import com.ibm.bi.dml.utils.DMLRuntimeException;

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
	
	
	public static MatrixBlock randOperationsOLD(int rows, int cols, double sparsity, double min, double max, String pdf, long seed)
	{
		MatrixBlock m = new MatrixBlock();
		
		if ( pdf.equalsIgnoreCase("normal") ) {
			m.getNormalRandomSparseMatrixOLD(rows, cols, sparsity, seed);
		}
		else {
			m.getRandomSparseMatrixOLD(rows, cols, sparsity, min, max, seed);
		}
		return m;
	}
	
	public static MatrixBlock randOperationsNEW(int rows, int cols, int rowsInBlock, int colsInBlock, double sparsity, double min, double max, String pdf, Well1024a bigrand) {
		MatrixBlock m = new MatrixBlock();
		if ( pdf.equalsIgnoreCase("normal") ) {
			m.getNormalRandomSparseMatrixNEW(rows, cols, rowsInBlock, colsInBlock, sparsity, bigrand, -1);
		}
		else {
			m.getRandomSparseMatrix(rows, cols, rowsInBlock, colsInBlock, sparsity, min, max, bigrand, -1);
		}
		return m;
	}
	
	public static MatrixBlock seqOperations(double from, double to, double incr) throws DMLRuntimeException {
		MatrixBlock m = new MatrixBlock();
		m.getSequence(from, to, incr);
		return m;
	}
}
