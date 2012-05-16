package com.ibm.bi.dml.test.integration;

import static junit.framework.Assert.assertTrue;
import static junit.framework.Assert.fail;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;

import org.junit.After;
import org.junit.Before;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.sql.sqlcontrolprogram.NetezzaConnector;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.ParameterBuilder;
import com.ibm.bi.dml.utils.Statistics;
import com.ibm.bi.dml.utils.configuration.DMLConfig;


/**
 * <p>
 * Extend this class to easily
 * </p>
 * <ul>
 * <li>set up an environment for DML script execution</li>
 * <li>use multiple test cases</li>
 * <li>generate test data></li>
 * <li>check results</li>
 * <li>clean up after test run</li>
 * </ul>
 * 
 */
public abstract class AutomatedTestBase {
	protected static final String SCRIPT_DIR = "./test/scripts/";
	protected static final String INPUT_DIR = "in/";
	protected static final String OUTPUT_DIR = "out/";
	protected static final String EXPECTED_DIR = "expected/";
	protected static final String TEMP_DIR = "./tmp/";
	protected static final boolean DEBUG = false;
	protected static final boolean RUNNETEZZA = false;
	
	protected static String baseDirectory;
	protected HashMap<String, TestConfiguration> availableTestConfigurations;
	
	/* For testing in the old way */
	protected HashMap<String, String> testVariables; /* variables and their values */

	/* For testing in the new way */
	protected String[] dmlArgs;            /* args to DMLScript.main */
	protected String[] dmlArgsDebug;       /* args to DMLScript.main with -d option */
	protected String rCmd;                 /* Rscript foo.R arg1, arg2 ...          */
	
	protected String selectedTest;
	protected String[] outputDirectories;
	protected String[] comparisonFiles;
	protected ArrayList<String> inputDirectories;
	protected ArrayList<String> inputRFiles;
	protected ArrayList<String> expectedFiles;

	private boolean isOutAndExpectedDeletionDisabled = false;
	private long lTimeBeforeTest = 0;

	private String expectedStdOut;
	private int iExpectedStdOutState = 0;
	private PrintStream originalPrintStreamStd = null;

	@Before
	public abstract void setUp();

	/**
	 * <p>
	 * Adds a test configuration to the list of available test configurations.
	 * </p>
	 * 
	 * @param testName
	 *            test name
	 * @param config
	 *            test configuration
	 */
	protected void addTestConfiguration(String testName, TestConfiguration config) {
		availableTestConfigurations.put(testName, config);
	}

	
	@Before
	public final void setUpBase() {
		availableTestConfigurations = new HashMap<String, TestConfiguration>();
		testVariables = new HashMap<String, String>();
		inputDirectories = new ArrayList<String>();
		inputRFiles = new ArrayList<String>();
		expectedFiles = new ArrayList<String>();
		outputDirectories = new String[0];
		isOutAndExpectedDeletionDisabled = false;
		lTimeBeforeTest = System.currentTimeMillis();
	}

	/**
	 * <p>
	 * Returns a test configuration from the list of available configurations.
	 * If no configuration is added for the specified name, the test will fail.
	 * </p>
	 * 
	 * @param testName
	 *            test name
	 * @return test configuration
	 */
	protected TestConfiguration getTestConfiguration(String testName) {
		if (!availableTestConfigurations.containsKey(testName))
			fail("unable to load test configuration");

		return availableTestConfigurations.get(testName);
	}

	/**
	 * <p>
	 * Generates a random matrix with the specified characteristics and returns
	 * it as a two dimensional array.
	 * </p>
	 * 
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 * @param sparsity
	 *            sparsity
	 * @param seed
	 *            seed
	 * @return two dimensional array containing random matrix
	 */
	protected double[][] getRandomMatrix(int rows, int cols, double min, double max, double sparsity, long seed) {
		return TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed);
	}

	/**
	 * <p>
	 * Generates a random matrix with the specified characteristics which does
	 * not contain any zero values and returns it as a two dimensional array.
	 * </p>
	 * 
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 * @param seed
	 *            seed
	 * @return two dimensional array containing random matrix
	 */
	protected double[][] getNonZeroRandomMatrix(int rows, int cols, double min, double max, long seed) {
		return TestUtils.generateNonZeroTestMatrix(rows, cols, min, max, seed);
	}

	/**
	 * <p>
	 * Generates a random matrix with the specified characteristics and writes
	 * it to a file.
	 * </p>
	 * 
	 * @param name
	 *            directory name
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 * @param sparsity
	 *            sparsity
	 * @param seed
	 *            seed
	 */
	protected double[][] createRandomMatrix(String name, int rows, int cols, double min, double max, double sparsity,
			long seed) {
		return createRandomMatrix(name, rows, cols, min, max, sparsity, seed, false);
	}

	/**
	 * <p>
	 * Generates a random matrix with the specified characteristics and writes
	 * it to a file.
	 * </p>
	 * 
	 * @param name
	 *            directory name
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 * @param sparsity
	 *            sparsity
	 * @param seed
	 *            seed
	 * @param bIncludeR
	 *            If true, writes also a R matrix to disk
	 */
	protected double[][] createRandomMatrix(String name, int rows, int cols, double min, double max, double sparsity,
			long seed, boolean bIncludeR) {
		double[][] matrix = TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, seed);
		String completePath = baseDirectory + INPUT_DIR + name + "/in";

		TestUtils.writeTestMatrix(completePath, matrix, bIncludeR);
		if (DEBUG)
			TestUtils.writeTestMatrix(TEMP_DIR + completePath, matrix);
		inputDirectories.add(baseDirectory + INPUT_DIR + name);
		return matrix;
	}

	/**
	 * <p>
	 * Generates a random matrix with the specified characteristics and writes
	 * it to a file.
	 * </p>
	 * 
	 * @param matrix
	 *            matrix characteristics
	 */
	protected void createRandomMatrix(TestMatrixCharacteristics matrix) {
		createRandomMatrix(matrix.getMatrixName(), matrix.getRows(), matrix.getCols(), matrix.getMinValue(), matrix
				.getMaxValue(), matrix.getSparsity(), matrix.getSeed());
	}

	/**
	 * <p>
	 * Adds a matrix to the input path and writes it to a file.
	 * </p>
	 * 
	 * @param name
	 *            directory name
	 * @param matrix
	 *            two dimensional matrix
	 * @param bIncludeR
	 *            generates also the corresponding R matrix
	 */
	protected double[][] writeInputMatrix(String name, double[][] matrix, boolean bIncludeR) {
		String completePath = baseDirectory + INPUT_DIR + name + "/in";
		String completeRPath = baseDirectory + INPUT_DIR + name + ".mtx";
		TestUtils.writeTestMatrix(completePath, matrix);
		if (bIncludeR) {
			TestUtils.writeTestMatrix(completeRPath, matrix, true);
			inputRFiles.add(completeRPath);
		}
		if (DEBUG)
			TestUtils.writeTestMatrix(TEMP_DIR + completePath, matrix);
		inputDirectories.add(baseDirectory + INPUT_DIR + name);

		return matrix;
	}

	protected double[][] createTable(String name, double[][] matrix) throws ClassNotFoundException
	{
		NetezzaConnector con = null;
		try{
		con = new NetezzaConnector();
		
		con.connect();
		con.createTable(baseDirectory + INPUT_DIR + name, matrix);
		
		return matrix;
		}
		catch(SQLException e)
		{
			return null;
		}
		finally
		{
			try
			{
				if(con != null)
					con.disconnect();
			}
			catch(Exception e)
			{
				e.printStackTrace();
			}
		}
	}
	/**
	 * <p>
	 * Adds a matrix to the input path and writes it to a file.
	 * </p>
	 * 
	 * @param name
	 *            directory name
	 * @param matrix
	 *            two dimensional matrix
	 */
	protected double[][] writeInputMatrix(String name, double[][] matrix) {
		return writeInputMatrix(name, matrix, false);
	}

	/**
	 * <p>
	 * Adds a matrix to the input path and writes it to a file in binary format.
	 * </p>
	 * 
	 * @param name
	 *            directory name
	 * @param matrix
	 *            two dimensional matrix
	 * @param rowsInBlock
	 *            rows in block
	 * @param colsInBlock
	 *            columns in block
	 * @param sparseFormat
	 *            sparse format
	 */
	protected void writeInputBinaryMatrix(String name, double[][] matrix, int rowsInBlock, int colsInBlock,
			boolean sparseFormat) {
		String completePath = baseDirectory + INPUT_DIR + name + "/in";
		if (rowsInBlock == 1 && colsInBlock == 1) {
			TestUtils.writeBinaryTestMatrixCells(completePath, matrix);
			if (DEBUG)
				TestUtils.writeBinaryTestMatrixCells(TEMP_DIR + completePath, matrix);
		} else {
			TestUtils.writeBinaryTestMatrixBlocks(completePath, matrix, rowsInBlock, colsInBlock, sparseFormat);
			if (DEBUG)
				TestUtils.writeBinaryTestMatrixBlocks(TEMP_DIR + completePath, matrix, rowsInBlock, colsInBlock,
						sparseFormat);
		}
		inputDirectories.add(baseDirectory + INPUT_DIR + name);
	}

	/**
	 * <p>
	 * Adds a matrix to the expectation path and writes it to a file.
	 * </p>
	 * 
	 * @param name
	 *            directory name
	 * @param matrix
	 *            two dimensional matrix
	 */
	protected void writeExpectedMatrix(String name, double[][] matrix) {
		TestUtils.writeTestMatrix(baseDirectory + EXPECTED_DIR + name, matrix);
		expectedFiles.add(baseDirectory + EXPECTED_DIR + name);
	}

	/**
	 * <p>
	 * Adds a matrix to the expectation path and writes it to a file in binary
	 * format.
	 * </p>
	 * 
	 * @param name
	 *            directory name
	 * @param matrix
	 *            two dimensional matrix
	 * @param rowsInBlock
	 *            rows in block
	 * @param colsInBlock
	 *            columns in block
	 * @param sparseFormat
	 *            sparse format
	 */
	protected void writeExpectedBinaryMatrix(String name, double[][] matrix, int rowsInBlock, int colsInBlock,
			boolean sparseFormat) {
		if (rowsInBlock == 1 && colsInBlock == 1) {
			TestUtils.writeBinaryTestMatrixCells(baseDirectory + EXPECTED_DIR + name + "/in", matrix);
		} else {
			TestUtils.writeBinaryTestMatrixBlocks(baseDirectory + EXPECTED_DIR + name + "/in", matrix, rowsInBlock,
					colsInBlock, sparseFormat);
		}
		inputDirectories.add(baseDirectory + EXPECTED_DIR + name);
	}

	/**
	 * <p>
	 * Creates a helper matrix which can be used for writing scalars to a file.
	 * </p>
	 */
	protected void createHelperMatrix() {
		TestUtils.writeTestMatrix(baseDirectory + INPUT_DIR + "helper/in", new double[][] { { 1, 1 } });
		inputDirectories.add(baseDirectory + INPUT_DIR + "helper");
	}

	/**
	 * <p>
	 * Creates a expectation helper matrix which can be used to compare scalars.
	 * </p>
	 * 
	 * @param name
	 *            file name
	 * @param value
	 *            scalar value
	 */
	protected void writeExpectedHelperMatrix(String name, double value) {
		TestUtils.writeTestMatrix(baseDirectory + EXPECTED_DIR + name, new double[][] { { value, value } });
		expectedFiles.add(baseDirectory + EXPECTED_DIR + name);
	}

	protected void writeExpectedScalar(String name, double value) {
		TestUtils.writeTestScalar(baseDirectory + EXPECTED_DIR + name, value);
		expectedFiles.add(baseDirectory + EXPECTED_DIR + name);
	}
	
	protected static HashMap<CellIndex, Double> readDMLMatrixFromHDFS(String fileName) {
		return TestUtils.readDMLMatrixFromHDFS(baseDirectory + OUTPUT_DIR + fileName);
	}

	public static HashMap<CellIndex, Double> readRMatrixFromFS(String fileName) {
		System.out.println("R script out: " + baseDirectory + EXPECTED_DIR + fileName);
		return TestUtils.readRMatrixFromFS(baseDirectory + EXPECTED_DIR + fileName);
	}
	
	protected static HashMap<CellIndex, Double> readDMLScalarFromHDFS(String fileName) {
		return TestUtils.readDMLScalarFromHDFS(baseDirectory + OUTPUT_DIR + fileName);
	}

	public static HashMap<CellIndex, Double> readRScalarFromFS(String fileName) {
		System.out.println("R script out: " + baseDirectory + EXPECTED_DIR + fileName);
		return TestUtils.readRScalarFromFS(baseDirectory + EXPECTED_DIR + fileName);
	}
	
	/**
	 * <p>
	 * Loads a test configuration with its parameters. Adds the output
	 * directories to the output list as well as to the list of possible
	 * comparison files.
	 * </p>
	 * 
	 * @param configurationName
	 *            test configuration name
	 * @author Felix Hamborg
	 */
	protected void loadTestConfiguration(TestConfiguration config) {
		if (!availableTestConfigurations.containsValue(config))
			fail("test configuration not available: " + config.getTestScript());
		String testDirectory = config.getTestDirectory();
		if (testDirectory != null)
			baseDirectory = SCRIPT_DIR + testDirectory;

		selectedTest = config.getTestScript();

		String[] outputFiles = config.getOutputFiles();
		outputDirectories = new String[outputFiles.length];
		comparisonFiles = new String[outputFiles.length];
		for (int i = 0; i < outputFiles.length; i++) {
			outputDirectories[i] = baseDirectory + OUTPUT_DIR + outputFiles[i];
			comparisonFiles[i] = baseDirectory + EXPECTED_DIR + outputFiles[i];
		}

		testVariables = config.getVariables();
		testVariables.put("basedir", baseDirectory);
		testVariables.put("indir", baseDirectory + INPUT_DIR);
		testVariables.put("outdir", baseDirectory + OUTPUT_DIR);
		testVariables.put("readhelper", "Helper = read(\"" + baseDirectory + INPUT_DIR + "helper/in\", "
				+ "rows=1, cols=2, format=\"text\");");
		testVariables.put("Routdir", baseDirectory + EXPECTED_DIR);

		if (DEBUG)
			TestUtils.clearDirectory(TEMP_DIR + baseDirectory + INPUT_DIR);
	}

	/**
	 * <p>
	 * Loads a test configuration with its parameters. Adds the output
	 * directories to the output list as well as to the list of possible
	 * comparison files.
	 * </p>
	 * 
	 * @param configurationName
	 *            test configuration name
	 * @author schnetter
	 */
	@Deprecated
	protected void loadTestConfiguration(String configurationName) {
		if (!availableTestConfigurations.containsKey(configurationName))
			fail("test configuration not available: " + configurationName);

		TestConfiguration config = availableTestConfigurations.get(configurationName);

		loadTestConfiguration(config);
	}

	/** 
	 * Runs an R script, default to the old way
	 */
	protected void runRScript() {
		runRScript(false);
		
	}
	/**
	 * Runs an R script in the old or the new way
	 */
	protected void runRScript(boolean newWay) {
	
		String executionFile = baseDirectory + selectedTest + ".R"; 
		
		String cmd;
		if (newWay == false) {
			executionFile = executionFile + "t";
			cmd = "R -f " + executionFile;
		}
		else {
			cmd = rCmd;
		}
		
		if (System.getProperty("os.name").contains("Windows")) {
			cmd = cmd.replace('/', '\\');                        
			executionFile = executionFile.replace('/', '\\');
		}
		
		if (newWay == false) {
		ParameterBuilder.setVariablesInScript(baseDirectory, selectedTest + ".R", testVariables);
		}
	
		if (DEBUG);
			TestUtils.printRScript(executionFile);
			
		try {
			System.out.println("starting R script");
			System.out.println("cmd: " + cmd);           
			Process child = Runtime.getRuntime().exec(cmd);     
			String outputR = "";
			int c = 0;

			while ((c = child.getInputStream().read()) != -1) {
				System.out.print((char) c);
				outputR += String.valueOf((char) c);
			}
			while ((c = child.getErrorStream().read()) != -1) {
				System.err.print((char) c);
			}

			/**
			 * To give any stream enough time to print all data, otherwise there
			 * are situations where the test case fails, even before everything
			 * has been printed
			 */
			child.waitFor();
	//		Thread.sleep(30000);

			try {
				if (child.exitValue() != 0) {
					throw new Exception("ERROR: R has ended irregularly\n" + outputR + "\nscript file: "
							+ executionFile);
				}
			} catch (IllegalThreadStateException ie) {
				/**
				 * In UNIX JVM does not seem to be able to close threads
				 * correctly. However, give it a try, since R processed the
				 * script, therefore we can terminate the process.
				 */
				child.destroy();
			}

			System.out.println("R is finished");

		} catch (Exception e) {
			e.printStackTrace();
			StringBuilder errorMessage = new StringBuilder();
			errorMessage.append("failed to run script " + executionFile);
			errorMessage.append("\nexception: " + e.toString());
			errorMessage.append("\nmessage: " + e.getMessage());
			errorMessage.append("\nstack trace:");
			for (StackTraceElement ste : e.getStackTrace()) {
				errorMessage.append("\n>" + ste);
			}
			fail(errorMessage.toString());
		}
	}

	/**
	 * <p>
	 * Runs a test for which no exception is expected.
	 * </p>
	 */
	protected void runTest() {
		runTest(false, null);
	}

	/**
	 * <p>
	 * Runs a test for which no exception is expected. If SystemML executes more
	 * MR jobs than specified in maxMRJobs this test will fail.
	 * </p>
	 * 
	 * @param maxMRJobs
	 *            specifies a maximum limit for the number of MR jobs. If set to
	 *            -1 there is no limit.
	 */
	protected void runTest(int maxMRJobs) {
		runTest(false, null, maxMRJobs);
	}

	/**
	 * <p>
	 * Runs a test for which the exception expectation can be specified.
	 * </p>
	 * 
	 * @param exceptionExpected
	 *            exception expected
	 */
	protected void runTest(boolean exceptionExpected) {
		runTest(exceptionExpected, null);
	}

	/**
	 * <p>
	 * Runs a test for which the exception expectation can be specified as well
	 * as the specific expectation which is expected.
	 * </p>
	 *
	 * @param exceptionExpected
	 *            exception expected
	 * @param expectedException
	 *            expected exception
	 */
	protected void runTest(boolean exceptionExpected, Class<?> expectedException) {
		runTest(exceptionExpected, expectedException, -1);
	}
	
	/**
	 * <p>
	 * Runs a test for which the exception expectation can be specified as well
	 * as the specific expectation which is expected. If SystemML executes more
	 * MR jobs than specified in maxMRJobs this test will fail.
	 * </p>
	 *
	 * @param exceptionExpected
	 *            exception expected
	 * @param expectedException
	 *            expected exception
	 * @param maxMRJobs
	 *            specifies a maximum limit for the number of MR jobs. If set to
	 *            -1 there is no limit.
	 */
	protected void runTest(boolean exceptionExpected, Class<?> expectedException, int maxMRJobs) {
		runTest(false, exceptionExpected, expectedException, maxMRJobs);
	}
		
	/**
	 * <p>
	 * Runs a test for which the exception expectation can be specified as well
	 * as the specific expectation which is expected. If SystemML executes more
	 * MR jobs than specified in maxMRJobs this test will fail.
	 * </p>
	 * @param newWay
	 * 			  in the new way if it is set to true
	 * @param exceptionExpected
	 *            exception expected
	 * @param expectedException
	 *            expected exception
	 * @param maxMRJobs
	 *            specifies a maximum limit for the number of MR jobs. If set to
	 *            -1 there is no limit.
	 */
	protected void runTest(boolean newWay, boolean exceptionExpected, Class<?> expectedException, int maxMRJobs) {
		
		String executionFile = baseDirectory + selectedTest + ".dml";
		
		if (newWay == false) {
			executionFile = executionFile + "t";
			ParameterBuilder.setVariablesInScript(baseDirectory, selectedTest + ".dml", testVariables);
		}
		
		//cleanup scratch folder (prevent side effect between tests)
		cleanupScratchSpace();
			
		if (DEBUG)
			TestUtils.printDMLScript(executionFile);
		
		try {
				if (newWay == false) {
					if (DEBUG)
						DMLScript.main(new String[] { "-f" ,executionFile, "-d" });
					else
						DMLScript.main(new String[] { "-f", executionFile });
				}
				else {
					if (DEBUG)
						DMLScript.main(dmlArgsDebug);
					else
						DMLScript.main(dmlArgs);
				}
		
			/** check number of MR jobs */
			if (maxMRJobs > -1 && maxMRJobs < Statistics.getNoOfCompiledMRJobs())
				fail("Limit of MR jobs exceeded: expected: " + maxMRJobs + ", occured: "
						+ Statistics.getNoOfCompiledMRJobs());

			if (exceptionExpected)
				fail("expected exception which has not been raised: " + expectedException);
		} catch (Exception e) {
			if (!exceptionExpected || (expectedException != null && !e.getClass().equals(expectedException))) {
				e.printStackTrace();
				StringBuilder errorMessage = new StringBuilder();
				errorMessage.append("failed to run script " + executionFile);
				errorMessage.append("\nexception: " + e.toString());
				errorMessage.append("\nmessage: " + e.getMessage());
				errorMessage.append("\nstack trace:");
				for (StackTraceElement ste : e.getStackTrace()) {
					errorMessage.append("\n>" + ste);
				}
				fail(errorMessage.toString());
			}
		}
	}
	
	protected void runSQL()
	{
		String executionFile = baseDirectory + selectedTest + ".dmlt";
		
		TestUtils.setVariablesInScript(baseDirectory, selectedTest + ".dml", testVariables);
		
		try {
		if(DEBUG)
			DMLScript.main(new String[] { "-f" ,executionFile, "-d", "-nz" });
		else
			DMLScript.main(new String[] { "-f" ,executionFile, "-nz" });
		} catch (Exception e) {
				e.printStackTrace();
				StringBuilder errorMessage = new StringBuilder();
				errorMessage.append("failed to run script " + executionFile);
				errorMessage.append("\nexception: " + e.toString());
				errorMessage.append("\nmessage: " + e.getMessage());
				errorMessage.append("\nstack trace:");
				for (StackTraceElement ste : e.getStackTrace()) {
					errorMessage.append("\n>" + ste);
				}
				fail(errorMessage.toString());
		}
	}
	
	protected void cleanupScratchSpace()
	{
		try 
		{
			//parse config file
			DMLConfig conf = new DMLConfig(DMLScript.DEFAULT_SYSTEMML_CONFIG_FILEPATH);

			// delete the scratch_space and all contents
			// (prevent side effect between tests)
			String dir = conf.getTextValue("scratch");  
			MapReduceTool.deleteFileIfExistOnHDFS(dir);
		} 
		catch (Exception ex) 
		{
			//ex.printStackTrace();
			return; //no effect on tests
		}
	}
	
	protected HashMap<CellIndex, Double> readDMLmatrixFromTable(String tableName)
	{
		NetezzaConnector con = new NetezzaConnector();
		try {
			con.connect();
			HashMap<CellIndex, Double> res = con.tableToHashMap(baseDirectory + OUTPUT_DIR + tableName);
			con.disconnect();
			return res;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}
	
	/**
	 * <p>
	 * Compares the results of the computation with the expected ones.
	 * </p>
	 */
	protected void compareResults() {
		compareResults(0);
	}

	/**
	 * <p>
	 * Compares the results of the computation with the expected ones with a
	 * specified tolerance.
	 * </p>
	 * 
	 * @param epsilon
	 *            tolerance
	 */
	protected void compareResultsWithR(double epsilon) {
		for (int i = 0; i < comparisonFiles.length; i++) {
			TestUtils.compareDMLHDFSFileWithRFile(comparisonFiles[i], outputDirectories[i], epsilon);
		}
	}

	/**
	 * <p>
	 * Compares the results of the computation with the Result calculated by R
	 * </p>
	 */
	protected void compareResultsWithR() {
		compareResultsWithR(0);
	}

	/**
	 * <p>
	 * Compares the results of the computation with the expected ones with a
	 * specified tolerance.
	 * </p>
	 * 
	 * @param epsilon
	 *            tolerance
	 */
	protected void compareResults(double epsilon) {
		for (int i = 0; i < comparisonFiles.length; i++) {
			/* Note that DML scripts may generate a file with only scalar value */
			if (outputDirectories[i].endsWith(".scalar")) {
			   String javaFile = comparisonFiles[i].replace(".scalar", "");
			   String dmlFile = outputDirectories[i].replace(".scalar", "");
			   TestUtils.compareDMLScalarWithJavaScalar(javaFile, dmlFile, epsilon);
			}
			else {
				TestUtils.compareDMLMatrixWithJavaMatrix(comparisonFiles[i], outputDirectories[i], epsilon);
			}
		}
	}
	
	
	/**
	 * Compare results of the computation with the expected results where rows may be permuted.
	 * @param epsilon
	 */
	protected void compareResultsRowsOutOfOrder(double epsilon)
	{
		for (int i = 0; i < comparisonFiles.length; i++) {
			/* Note that DML scripts may generate a file with only scalar value */
			if (outputDirectories[i].endsWith(".scalar")) {
			   String javaFile = comparisonFiles[i].replace(".scalar", "");
			   String dmlFile = outputDirectories[i].replace(".scalar", "");
			   TestUtils.compareDMLScalarWithJavaScalar(javaFile, dmlFile, epsilon);
			}
			else {
				TestUtils.compareDMLMatrixWithJavaMatrixRowsOutOfOrder(comparisonFiles[i], outputDirectories[i], epsilon);
			}
		}
	}

	/**
	 * <p>
	 * Compares the results of the computation with the expected ones which can
	 * be in different order in the file.
	 * </p>
	 * 
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 */
	@Deprecated
	protected void compareResultsInDifferentOrder(int rows, int cols) {
		compareResultsInDifferentOrder(rows, cols, 0);
	}

	/**
	 * <p>
	 * Compares the results of the computation with the expected ones which can
	 * be in different order in the file with a specified tolerance.
	 * </p>
	 * 
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param epsilon
	 *            tolerance
	 */
	@Deprecated
	protected void compareResultsInDifferentOrder(int rows, int cols, double epsilon) {
		for (int i = 0; i < comparisonFiles.length; i++) {
			TestUtils.compareFilesInDifferentOrder(comparisonFiles[i], outputDirectories[i], rows, cols, epsilon);
		}
	}

	/**
	 * <p>
	 * Checks the results of a computation against a number of characteristics.
	 * </p>
	 * 
	 * @param rows
	 *            number of rows
	 * @param cols
	 *            number of columns
	 * @param min
	 *            minimum value
	 * @param max
	 *            maximum value
	 */
	protected void checkResults(long rows, long cols, double min, double max) {
		for (int i = 0; i < outputDirectories.length; i++) {
			TestUtils.checkMatrix(outputDirectories[i], rows, cols, min, max);
		}
	}

	/**
	 * <p>
	 * Checks for the existence for all of the outputs.
	 * </p>
	 */
	protected void checkForResultExistence() {
		for (int i = 0; i < outputDirectories.length; i++) {
			TestUtils.checkForOutputExistence(outputDirectories[i]);
		}
	}

	@After
	public void tearDown() {
		System.out.println("Duration: " + (System.currentTimeMillis() - lTimeBeforeTest) + "ms");

		assertTrue("expected String did not occur: " + expectedStdOut, iExpectedStdOutState == 0
				|| iExpectedStdOutState == 2);
		TestUtils.displayAssertionBuffer();

		TestUtils.removeHDFSDirectories(inputDirectories.toArray(new String[inputDirectories.size()]));
		TestUtils.removeFiles(inputRFiles.toArray(new String[inputRFiles.size()]));
		TestUtils.removeTemporaryFiles();

		if (!isOutAndExpectedDeletionDisabled) {
			TestUtils.clearDirectory(baseDirectory + OUTPUT_DIR);
			TestUtils.removeHDFSFiles(expectedFiles.toArray(new String[expectedFiles.size()]));
			TestUtils.clearDirectory(baseDirectory + EXPECTED_DIR);
			TestUtils.removeFiles(new String[] { baseDirectory + selectedTest + ".dmlt" });
			TestUtils.removeFiles(new String[] { baseDirectory + selectedTest + ".Rt" });
		}

		TestUtils.clearAssertionInformation();

		System.gc();
	}

	/**
	 * Disables the deletion of files and directories in the output and expected
	 * folder for this test.
	 */
	public void disableOutAndExpectedDeletion() {
		isOutAndExpectedDeletionDisabled = true;
	}

	/**
	 * Enables expection of a line in standard output stream.
	 * 
	 * @param expected
	 */
	public void setExpectedStdOut(String expectedLine) {
		this.expectedStdOut = expectedLine;
		originalPrintStreamStd = System.out;
		iExpectedStdOutState = 1;
		System.setOut(new PrintStream(new ExpecterOutputStream()));
	}

	/**
	 * This class is used to compare the standard output stream against an
	 * expected string.
	 * 
	 * @author Felix Hamborg
	 * 
	 */
	class ExpecterOutputStream extends OutputStream {
		private String line = "";

		@Override
		public void write(int b) throws IOException {
			line += String.valueOf((char) b);
			if (((char) b) == '\n') {
				/** new line */
				if (line.contains(expectedStdOut)) {
					iExpectedStdOutState = 2;
					System.setOut(originalPrintStreamStd);
				}
			}
			originalPrintStreamStd.write(b);
		}
	}

	/**
	 * <p>
	 * Generates a matrix containing easy to debug values in its cells.
	 * </p>
	 * 
	 * @param rows
	 * @param cols
	 * @param bContainsZeros
	 *            If true, the matrix contains zeros. If false, the matrix
	 *            contains only positive values.
	 * @return
	 */
	protected double[][] createNonRandomMatrixValues(int rows, int cols, boolean bContainsZeros) {
		return TestUtils.createNonRandomMatrixValues(rows, cols, bContainsZeros);
	}

	/**
	 * <p>
	 * Generates a matrix containing easy to debug values in its cells. The
	 * generated matrix contains zero values
	 * </p>
	 * 
	 * @param rows
	 * @param cols
	 * @return
	 */
	protected double[][] createNonRandomMatrixValues(int rows, int cols) {
		return TestUtils.createNonRandomMatrixValues(rows, cols, true);
	}
}
