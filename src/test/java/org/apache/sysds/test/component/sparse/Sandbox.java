package org.apache.sysds.test.component.sparse;

import org.apache.sysds.runtime.data.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlockMCSC;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Sandbox extends AutomatedTestBase {

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void myTest(){
		tester(SparseBlock.Type.MCSR);

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

			SparseBlockCSC newBlock = new SparseBlockCSC(sblock);

			newBlock.reset(1, 3, 3);

			//System.out.println(newBlock.size(1));
			System.out.println(newBlock.size());


			System.out.println("values:");
			double[] vals = newBlock.values(0);
			for(double val : vals)
				System.out.println(val);

			System.out.println("**********************");
			System.out.println("indexes");
			int[] indexes = newBlock.indexes(0);
			for(int idx: indexes)
				System.out.println(idx);

			System.out.println("***********************");
			System.out.println("pointers");
			for(int i = 0; i < 7; i++){
				System.out.println(newBlock.pos(i));
			}

			System.out.println("***********************");











		}catch(Exception ex){
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
