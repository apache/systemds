/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.test.integration.functions.io;

import java.io.IOException;

import org.junit.Assert;

import org.junit.Test;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class SeqParReadTest extends AutomatedTestBase {

	
	private final static String TEST_NAME = "SeqParReadTest";
	private final static String TEST_DIR = "functions/io/";
	
	private final static int rowsA = 2000;
	private final static int colsA = 1000;
	private final static int colsB = 7;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;
	
	private final static double eps = 1e-9;

	/**
	 * Main method for running one test at a time.
	 */
	public static void main(String[] args) {
		long startMsec = System.currentTimeMillis();

		SeqParReadTest t = new SeqParReadTest();
		t.setUpBase();
		t.setUp();
		t.testParReadTextcellSparseBig();
		t.tearDown();

		long elapsedMsec = System.currentTimeMillis() - startMsec;
		System.err.printf("Finished in %1.3f sec\n", elapsedMsec / 1000.0);

	}
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   ); 
	}
	
	@Test
	public void testSeqReadCSVSparseSmall() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.CSVOutputInfo, false, false);
	}

	@Test
	public void testSeqReadTextcellSparseSmall() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.TextCellOutputInfo, false, false);
	}
	
	@Test
	public void testSeqReadMMSparseSmall() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.MatrixMarketOutputInfo, false, false);
	}

	@Test
	public void testParReadCSVSparseSmall() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.CSVOutputInfo, false, false);
	}

	@Test
	public void testParReadTextcellSparseSmall() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.TextCellOutputInfo, false, false);
	}
	
	@Test
	public void testParReadMMSparseSmall() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.MatrixMarketOutputInfo, false, false);
	}
	
	@Test
	public void testSeqReadCSVDenseSmall() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.CSVOutputInfo, true, false);
	}

	@Test
	public void testSeqReadTextcellDenseSmall() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.TextCellOutputInfo, true, false);
	}
	
	@Test
	public void testSeqReadMMDenseSmall() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.MatrixMarketOutputInfo, true, false);
	}

	@Test
	public void testParReadCSVDenseSmall() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.CSVOutputInfo, true, false);
	}

	@Test
	public void testParReadTextcellDenseSmall() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.TextCellOutputInfo, true, false);
	}
	
	@Test
	public void testParReadMMDenseSmall() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.MatrixMarketOutputInfo, false, false);
	}
	
	@Test
	public void testSeqReadCSVSparseBig() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.CSVOutputInfo, false, true);
	}

	@Test
	public void testSeqReadTextcellSparseBig() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.TextCellOutputInfo, false, true);
	}
	
	@Test
	public void testSeqReadMMSparseBig() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.MatrixMarketOutputInfo, false, true);
	}

	@Test
	public void testParReadCSVSparseBig() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.CSVOutputInfo, false, true);
	}

	@Test
	public void testParReadTextcellSparseBig() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.TextCellOutputInfo, false, true);
	}
	
	@Test
	public void testParReadMMSparseBig() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.MatrixMarketOutputInfo, false, true);
	}
	
	@Test
	public void testSeqReadCSVDenseBig() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.CSVOutputInfo, true, true);
	}

	@Test
	public void testSeqReadTextcellDenseBig() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.TextCellOutputInfo, true, true);
	}
	
	@Test
	public void testSeqReadMMDenseBig() {
		runReadTypeFormatSparsitySizeTest(false, OutputInfo.MatrixMarketOutputInfo, true, true);
	}

	@Test
	public void testParReadCSVDenseBig() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.CSVOutputInfo, true, true);
	}

	@Test
	public void testParReadTextcellDenseBig() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.TextCellOutputInfo, true, true);
	}
	
	@Test
	public void testParReadMMDenseBig() {
		runReadTypeFormatSparsitySizeTest(true, OutputInfo.MatrixMarketOutputInfo, true, true);
	}

	/*
	 * Generate a matrix (big/small dense/sparse) and write it as AX(text/csv/mm/bin) & BX(mm)
	 * Read AX and sum-up using DML script scripts/functions/io/SeqParReadTest.dml
	 * Read BX and sum-up using RScript scripts/functions/io/matrixmarket/mm_verify.R in MM format
	 * Compare the two results
	 * 
	 * @param parallel : Is the read parallel or not(sequential)
	 * @param fmt : text/csv/mm
	 * @param dense : Is the matrix dense or not(sparse)
	 * @param big : Is the matrix big or not(small)
	 * 
	 */
	
	private void runReadTypeFormatSparsitySizeTest(boolean parallel, OutputInfo fmt, boolean dense, boolean big ) {
		
		boolean oldpar = OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS;

		try
		{
			OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS = parallel;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rowsA, big?colsA:colsB, 0, 1, dense?sparsity2:sparsity1, 7); 
			writeMatrix(A, HOME + INPUT_DIR + "AX", fmt, rowsA, big?colsA:colsB, 1000, 1000, rowsA*(big?colsA:colsB));
			
			//always write in MM format for R
			writeMatrix(A, HOME + INPUT_DIR + "BX", OutputInfo.MatrixMarketOutputInfo, rowsA, big?colsA:colsB, 1000, 1000, rowsA*(big?colsA:colsB));
			
			String dmlOutput = HOME + OUTPUT_DIR + "dml.scalar";
			String rOutput = HOME + OUTPUT_DIR + "R.scalar";
			
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "AX", dmlOutput};
			
			fullRScriptName = HOME + "matrixmarket/mm_verify.R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + HOME + INPUT_DIR + "BX " + rOutput;
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			double dmlScalar = TestUtils.readDMLScalar(dmlOutput); 
			double rScalar = TestUtils.readRScalar(rOutput); 
			
			TestUtils.compareScalars(dmlScalar, rScalar, eps);
		} catch (Exception e) {
			e.printStackTrace();
			Assert.fail();
		}
		finally
		{
			OptimizerUtils.PARALLEL_CP_READ_TEXTFORMATS = oldpar;		
		}
	}
	
	
	/**
	 * 
	 * @param A
	 * @param dir
	 * @param oi
	 * @param rows
	 * @param cols
	 * @param brows
	 * @param bcols
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private void writeMatrix( double[][] A, String fname, OutputInfo oi, long rows, long cols, int brows, int bcols, long nnz ) 
		throws DMLRuntimeException, IOException
	{
		MapReduceTool.deleteFileIfExistOnHDFS(fname);
		MapReduceTool.deleteFileIfExistOnHDFS(fname+".mtd");
		
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, brows, bcols, nnz);
		MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
		DataConverter.writeMatrixToHDFS(mb, fname, oi, mc);
		if( oi != OutputInfo.MatrixMarketOutputInfo )
			MapReduceTool.writeMetaDataFile(fname+".mtd", ValueType.DOUBLE, mc, oi);
	}
}
