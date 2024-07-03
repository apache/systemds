package org.apache.sysds.test.component.sparse;

import org.apache.sysds.api.mlcontext.Matrix;
import org.apache.sysds.runtime.data.*;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

import java.util.Iterator;

public class Sandbox extends AutomatedTestBase{


	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}


	@Test
	public void runtest(){
		testMethod(SparseBlock.Type.MCSR);
	}




	private void testMethod(SparseBlock.Type btype){

		double[][] C = {{10, 20, 0, 0, 0, 0},
						{0, 30, 0, 40, 0, 0},
						{0, 0, 50, 60, 70, 0},
						{0, 0, 0, 0, 0, 80}};
		int rows = 10;
		int cols = 10;
		int rl = 0;
		int ru = 4;
		int cl = 0;
		int cu = 6;
		double sparsity = 0.1;

		SparseBlock sblock = null;
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(C);
		final int clen = mbtmp.getNumColumns();
		//System.out.println(clen);
		SparseBlock srtmp = mbtmp.getSparseBlock();
		switch(btype) {
			case MCSR:
				sblock = new SparseBlockMCSR(srtmp);
				break;
			case CSR:
				sblock = new SparseBlockCSR(srtmp);
				break;
			case COO:
				sblock = new SparseBlockCOO(srtmp);
				break;
			case DCSR:
				sblock = new SparseBlockDCSR(srtmp);
				break;
		}

		//System.out.println(sblock);
		SparseBlock newBlock = new SparseBlockMCSC(sblock);
		//System.out.println(newBlock);

		Iterator<IJV> iter = newBlock.getIterator();
		int count = 0;
		while( iter.hasNext() ) {
			System.out.println("here");
			IJV cell = iter.next();
			System.out.println(cell.getV());
		}





		double[][] B1 = {{1, 1, 0, 0, 0, 0},
						 {0, 0, 0, 0, 0, 0},
						 {0, 0, 1, 0, 0, 0},
						 {1, 0, 0, 0, 0, 0}};

		double[][] B2 = {{0, 0, 0, 0, 0, 0},
						 {0, 2, 0, 0, 0, 0},
						 {0, 0, 0, 0, 0, 0},
						 {2, 0, 0, 0, 0, 2}};










	}
}
