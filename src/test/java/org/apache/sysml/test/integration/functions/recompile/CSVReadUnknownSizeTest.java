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

package org.apache.sysml.test.integration.functions.recompile;

import java.util.HashMap;
import org.junit.Test;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class CSVReadUnknownSizeTest extends AutomatedTestBase {

	private final static String TEST_NAME = "csv_read_unknown";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + CSVReadUnknownSizeTest.class.getSimpleName() + "/";

	private final static int rows = 10;
	private final static int cols = 15;

	/** Main method for running one test at a time from Eclipse. */
	public static void main(String[] args) {
		long startMsec = System.currentTimeMillis();

		CSVReadUnknownSizeTest t = new CSVReadUnknownSizeTest();
		t.setUpBase();
		t.setUp();
		t.testCSVReadUnknownSizeSplitRewrites();
		t.tearDown();

		long elapsedMsec = System.currentTimeMillis() - startMsec;
		System.err.printf("Finished in %1.3f sec\n", elapsedMsec / 1000.0);
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "X" }));
	}

	@Test
	public void testCSVReadUnknownSizeNoSplitNoRewrites() {
		runCSVReadUnknownSizeTest(false, false);
	}

	@Test
	public void testCSVReadUnknownSizeNoSplitRewrites() {
		runCSVReadUnknownSizeTest(false, true);
	}

	@Test
	public void testCSVReadUnknownSizeSplitNoRewrites() {
		runCSVReadUnknownSizeTest(true, false);
	}

	@Test
	public void testCSVReadUnknownSizeSplitRewrites() {
		runCSVReadUnknownSizeTest(true, true);
	}

	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 */
	private void runCSVReadUnknownSizeTest( boolean splitDags, boolean rewrites )
	{	
		boolean oldFlagSplit = OptimizerUtils.ALLOW_SPLIT_HOP_DAGS;
		boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("X"), output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			OptimizerUtils.ALLOW_SPLIT_HOP_DAGS = splitDags;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			double[][] X = getRandomMatrix(rows, cols, -1, 1, 1.0d, 7);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(X);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, 1000, 1000);
			CSVFileFormatProperties fprop = new CSVFileFormatProperties();			
			DataConverter.writeMatrixToHDFS(mb, input("X"), OutputInfo.CSVOutputInfo, mc, -1, fprop);
			mc.set(-1, -1, -1, -1);
			MapReduceTool.writeMetaDataFile(input("X.mtd"), ValueType.DOUBLE, mc, OutputInfo.CSVOutputInfo, fprop);
			
			runTest(true, false, null, -1); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
				{
					Double tmp = dmlfile.get(new CellIndex(i+1,j+1));
					
					double expectedValue = mb.quickGetValue(i, j);			
					double actualValue =  (tmp==null)?0.0:tmp;
					
					if (expectedValue != actualValue) {
						throw new Exception(String.format("Value of cell (%d,%d) "
								+ "(zero-based indices) in output file %s is %f, "
								+ "but original value was %f",
								i, j, baseDirectory + OUTPUT_DIR + "R",
								actualValue, expectedValue));
					}
				}
			
			
			//check expected number of compiled and executed MR jobs
			//note: with algebraic rewrites - unary op in reducer prevents job-level recompile
			int expectedNumCompiled = (rewrites && !splitDags) ? 2 : 3; //reblock, GMR
			int expectedNumExecuted = splitDags ? 0 : rewrites ? 2 : 2;			
			
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
		finally
		{
			OptimizerUtils.ALLOW_SPLIT_HOP_DAGS = oldFlagSplit;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
		}
	}
}