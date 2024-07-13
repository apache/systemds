package org.apache.sysds.test.component.sparse;
import org.apache.sysds.runtime.data.*;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
public class Sandbox extends  AutomatedTestBase{

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void runtest()  {
		runSparseTest(SparseBlock.Type.MCSC);
	}

	private void runSparseTest( SparseBlock.Type btype)
	{
		try
		{
			//data generation
			//double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 1234);
			double[][] A = {{0, 0, 0, 0, 0, 0},
							{0, 0, 0, 0, 0, 0},
							{0, 0, 0, 0, 0, 0},
							{0, 0, 0, 0, 0, 80}};

			//init sparse block
			SparseBlock sblock = null;
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseBlock srtmp = mbtmp.getSparseBlock();
			switch( btype ) {
				case MCSR: sblock = new SparseBlockMCSR(srtmp); break;
				case CSR: sblock = new SparseBlockCSR(srtmp); break;
				case COO: sblock = new SparseBlockCOO(srtmp); break;
				case DCSR: sblock = new SparseBlockDCSR(srtmp); break;
				case MCSC: sblock = new SparseBlockMCSC(srtmp); break;
			}

			long nnz = sblock.size(0, 4, 0, 5);
			System.out.println(nnz);


		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}



}
