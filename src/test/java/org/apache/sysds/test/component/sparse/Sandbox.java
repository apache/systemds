package org.apache.sysds.test.component.sparse;

import org.apache.sysds.runtime.data.*;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

public class Sandbox extends AutomatedTestBase{



	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void myTest()  {
		runTest(SparseBlock.Type.MCSC);
	}





	private void runTest( SparseBlock.Type btype)
	{
		try
		{
			//data generation
			double[][] A = {{0, 20, 50, 0, 0, 0},
							{0, 30, 0, 40, 0, 0},
							{0, 0, 0, 60, 70, 0},
							{0, 0, 0, 0, 0, 0}};

			//init sparse block
			SparseBlock sblock = null;
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseBlock srtmp = mbtmp.getSparseBlock();
			switch( btype ) {
				case MCSR: sblock = new SparseBlockMCSR(srtmp); break;
				case CSR: sblock = new SparseBlockCSR(srtmp); break;
				case COO: sblock = new SparseBlockCOO(srtmp); break;
				case DCSR: sblock = new SparseBlockDCSR(srtmp); break;
				case MCSC: sblock = new SparseBlockMCSC(srtmp, 6); break;
			}

			//System.out.println(((SparseBlockMCSC)sblock).getCols()[0]);

			SparseRow row = sblock.get(3);
			System.out.println(row);











		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
