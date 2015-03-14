/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.external;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class SequenceMinerTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/external/";
	private final static String TEST_SEQMINER = "SeqMiner"; 
	
	/**
	 * Main method for running one test at a time.
	 */
	public static void main(String[] args) {
		long startMsec = System.currentTimeMillis();

		SequenceMinerTest t = new SequenceMinerTest();
		t.setUpBase();
		t.setUp();
		t.testSequenceMiner();
		t.tearDown();

		long elapsedMsec = System.currentTimeMillis() - startMsec;
		System.err.printf("Finished in %1.3f sec\n", elapsedMsec / 1000.0);

	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_SEQMINER, new TestConfiguration(TEST_DIR, TEST_SEQMINER, new String[] { "fseq", "sup"}));
	}

	@Test
	public void testSequenceMiner() {
		
		int rows = 5;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("SeqMiner");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		
		// Need to load the config here because only loadTestConfiguration sets baseDirectory
		loadTestConfiguration(config);

		/* This is for running the junit test by constructing the arguments directly */
		String SEQMINER_HOME = baseDirectory;
		fullDMLScriptName = SEQMINER_HOME + TEST_SEQMINER + ".dml";
		programArgs = new String[]{"-args",  SEQMINER_HOME + INPUT_DIR + "M" , 
				                        Integer.toString(rows), Integer.toString(cols), 
				                         SEQMINER_HOME + OUTPUT_DIR + "fseq" ,
				                         SEQMINER_HOME + OUTPUT_DIR + "sup" };
		
		double[][] M = {{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}, 
				{1, 2, 3, 4, -1, 2, 3, 4, 5, -1}};
		
		
		writeInputMatrix("M", M);
		
		HashMap<CellIndex, Double> fseq = TestUtils.readDMLMatrixFromHDFS(baseDirectory + "seqMiner/FreqSeqFile");
		HashMap<CellIndex, Double> sup = TestUtils.readDMLMatrixFromHDFS(baseDirectory + "seqMiner/FreqSeqSupportFile");
		
		
		double [][] expected_fseq = TestUtils.convertHashMapToDoubleArray(fseq);
		double [][] expected_sup = TestUtils.convertHashMapToDoubleArray(sup);
		
		
		
		writeExpectedMatrix("fseq", expected_fseq);
		writeExpectedMatrix("sup", expected_sup);
		


		// no expected number of M/R jobs are calculated, set to default for now
		runTest(true, false, null, -1);

		compareResultsRowsOutOfOrder(0.0);
	}
}
