package com.ibm.bi.dml.runtime.matrix.io;

import java.util.HashMap;

public class MatrixBlock extends MatrixBlockDSM{

	public MatrixBlock(int i, int j, boolean sparse1) {
		super(i, j, sparse1);
	}

	public MatrixBlock() {
		super();
	}
	
	public MatrixBlock(HashMap<CellIndex, Double> map) {
		super(map);
	}
	
	@Override
	public long getObjectSizeInMemory ()
	{
		return 0 + super.getObjectSizeInMemory ();
	}
}
