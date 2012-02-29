package dml.runtime.matrix.io;

public class MatrixBlock extends MatrixBlockDSM{

	public MatrixBlock(int i, int j, boolean sparse1) {
		super(i, j, sparse1);
	}

	public MatrixBlock() {
		super();
	}
}
