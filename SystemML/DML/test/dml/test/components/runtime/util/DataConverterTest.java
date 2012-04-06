package dml.test.components.runtime.util;

import static org.junit.Assert.fail;

import org.junit.Test;

import dml.runtime.util.DataConverter;
import dml.test.utils.TestUtils;
import dml.utils.DMLRuntimeException;

public class DataConverterTest {
	int blockCols = 200;
	int blockRows = 200;
	int cols = 300;
	int rows = 220;
	String matrix = "scripts/a";

/*	@Test
	public void testReadDouble1DArrayMatrixFromHDFSBlock() {
		double[][] arrIn = TestUtils.createNonRandomMatrixValues(rows, cols, false);
		double[] arrExp = TestUtils.convert2Dto1DDoubleArray(arrIn);

		TestUtils.writeBinaryTestMatrixBlocks(matrix, arrIn, blockRows, blockCols, true);
		try {
			double[] array = DataConverter.readDouble1DArrayMatrixFromHDFSBlock(matrix, rows, cols, blockRows,
					blockCols);
			for (int d = 0; d < arrExp.length; d++) {
				// System.out.println("E: " + arrExp[d] + "\tC: " + array[d]);
				if (arrExp[d] != array[d])
					fail("Arrays differ");
			}
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
*/
/*	@Test
	public void testReadDouble1DArrayMatrixFromHDFSCell() {
		double[][] arrIn = TestUtils.createNonRandomMatrixValues(rows, cols, false);
		double[] arrExp = TestUtils.convert2Dto1DDoubleArray(arrIn);
		
		TestUtils.writeBinaryTestMatrixCells(matrix, arrIn);

		try {
			double[] array = DataConverter.readDouble1DArrayMatrixFromHDFSCell(matrix, rows, cols);
			for (int d = 0; d < arrExp.length; d++) {
				if (arrExp[d] != array[d])
					fail("differ");
			}
		} catch (DMLRuntimeException e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
*/}
