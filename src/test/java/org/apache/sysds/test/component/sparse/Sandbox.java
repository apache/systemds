package org.apache.sysds.test.component.sparse;

import org.apache.sysds.runtime.data.*;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.IJV;
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
	public void runtest(){
		testMethod(SparseBlock.Type.MCSR);
	}




	private void testMethod(SparseBlock.Type btype){

		double[][] A = {{10, 20, 0, 0, 0, 0},
						{0, 30, 0, 40, 0, 0},
						{0, 0, 50, 60, 70, 0},
						{0, 0, 0, 0, 0, 80}};

		SparseBlock sblock = null;
		MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
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

		System.out.println(newBlock);

		System.out.println(((SparseBlockMCSR) sblock).getExactSizeInMemory());




	}
}
