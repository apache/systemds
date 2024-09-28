package org.apache.sysds.test.component.sparse;

import org.apache.sysds.runtime.data.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlockMCSC;
public class Sandbox extends AutomatedTestBase {

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void myTest(){
		tester(SparseBlock.Type.MCSC);

	}

	private void tester(SparseBlock.Type btype){
		try{

			double[][] A = {
				{10, 20, 0, 0, 0, 0},
				{0, 30, 0, 40, 0, 0},
				{0, 0, 50, 60, 70, 0},
				{0, 0, 0, 0, 0, 80},
			};

			SparseBlock sblock = null;
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			int rows= mbtmp.getNumRows();
			int cols= mbtmp.getNumColumns();
			SparseBlock srtmp = mbtmp.getSparseBlock();
			switch( btype ) {
				case MCSR: sblock = new SparseBlockMCSR(srtmp); break;
				case CSR: sblock = new SparseBlockCSR(srtmp); break;
				case COO: sblock = new SparseBlockCOO(srtmp); break;
				case DCSR: sblock = new SparseBlockDCSR(srtmp); break;
				case MCSC: sblock = new SparseBlockMCSC(srtmp, cols); break;
			}

			SparseBlockMCSC originalMCSC = (SparseBlockMCSC) sblock;
			SparseRow[] columns = originalMCSC.getCols();

			SparseBlockCSC cscBlock = new SparseBlockCSC(originalMCSC);
			int[] idx = cscBlock.indexesCol(0);
			double[] vals = cscBlock.valuesCol(0);


			/*for(int x : idx)
				System.out.println(x);

			System.out.println("**********");

			for(double x : vals)
				System.out.println(x);*/

			SparseBlockCSR csrBlock = new SparseBlockCSR(cscBlock);

			//System.out.println(csrBlock);


			System.out.println(sblock);






		}catch(Exception ex){
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
