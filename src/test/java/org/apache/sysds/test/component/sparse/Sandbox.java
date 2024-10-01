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



			/*SparseRow[] columns = new SparseRow[6];
			columns[0] = new SparseRowScalar(0, 10);

			double[] vals1 = {20, 30};
			int idx1[] = {0, 1};
			columns[1] = new SparseRowVector(vals1, idx1);

			columns[2] = new SparseRowScalar(2, 50);

			double[] vals3 = {40, 60};
			int idx3[] = {1, 2};
			columns[3] = new SparseRowVector(vals3, idx3);

			columns[4] = new SparseRowScalar(2, 70);

			columns[5] = new SparseRowScalar(3, 80);

			SparseBlockCSC newBlock = new SparseBlockCSC(columns, 8);*/

			/*double[] values = {10, 20, 30, 50, 40, 60, 70, 80};
			int rowInd[] = {0, 0, 1, 2, 1, 2, 2, 3};
			int colInd[] = {0, 1, 1, 2, 3, 3, 4, 5};
			//SparseBlockCSC newBlock = new SparseBlockCSC(6, rowInd, colInd, values);
			SparseBlockCSR newBlock = new SparseBlockCSR(6, 8, rowInd);

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

			System.out.println("***********************");*/


			// Initialize variables
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			DataOutputStream dos = new DataOutputStream(baos);
			int nnz = 0;
			int clen = A[0].length; // Number of columns

			// Traverse the matrix column-wise and write data in the expected format
			for (int j = 0; j < clen; j++) {
				int colNnz = 0;
				List<Integer> rowIndices = new ArrayList<>();
				List<Double> values = new ArrayList<>();

				// Collect non-zero elements in column j
				for (int i = 0; i < A.length; i++) {
					if (A[i][j] != 0.0) {
						rowIndices.add(i);
						values.add(A[i][j]);
						colNnz++;
						nnz++;
					}
				}

				// Write lnnz for column j
				dos.writeInt(colNnz);

				// Write row indices and values for column j
				for (int k = 0; k < colNnz; k++) {
					dos.writeInt(rowIndices.get(k)); // Row index
					dos.writeDouble(values.get(k));  // Value
				}
			}
			dos.flush();

			// Create DataInput from the ByteArrayOutputStream
			ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
			DataInput in = new DataInputStream(bais);


			newBlock.initSparse(6,nnz, in);

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
