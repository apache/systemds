/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.transform;

import static org.junit.Assert.assertTrue;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.io.ReaderBinaryBlock;
import com.ibm.bi.dml.runtime.io.ReaderTextCSV;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.transform.TransformationAgent.TX_METHOD;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.json.java.JSONArray;
import com.ibm.json.java.JSONObject;

/**
 * 
 * 
 */
public class ScalingTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "Scaling";

	private final static String TEST_DIR = "functions/transform/";
	
	private final static int rows1 = 1500;
	private final static int cols1 = 16;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_DIR, TEST_NAME,new String[]{"R"}));
	}

	
	@Test
	public void testTransformScalingCPCSV() throws IOException, DMLRuntimeException 
	{
		runScalingTest(rows1, cols1, ExecType.CP, "csv");
	}
	
	@Test
	public void testTransformScalingMRCSV() throws IOException, DMLRuntimeException 
	{
		runScalingTest(rows1, cols1, ExecType.MR, "csv");
	}
	
	@Test
	public void testTransformScalingCPBinary() throws IOException, DMLRuntimeException 
	{
		runScalingTest(rows1, cols1, ExecType.CP, "binary");
	}
	
	@Test
	public void testTransformScalingMRBinary() throws IOException, DMLRuntimeException 
	{
		runScalingTest(rows1, cols1, ExecType.MR, "binary");
	}
	
	private void generateSpecFile(int cols, String specFile) throws IOException 
	{
		final String NAME = "name";
		final String METHOD = "method";
		final String SCALE_METHOD_Z = "z-score";
		final String SCALE_METHOD_M = "mean-subtraction";
		
		JSONObject outputSpec = new JSONObject();
		JSONArray scaleSpec = new JSONArray();

		for(int colID=1; colID <= cols; colID++)
		{
			JSONObject obj = new JSONObject();
			obj.put(NAME, "V"+colID);
			if(colID <= cols/2)
				obj.put(METHOD, SCALE_METHOD_M);
			else
				obj.put(METHOD, SCALE_METHOD_Z);
			scaleSpec.add(obj);
		}
		outputSpec.put(TX_METHOD.SCALE.toString(), scaleSpec);
		
		FileSystem fs = FileSystem.get(TestUtils.conf);
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(specFile),true)));
		out.write(outputSpec.toString());
		out.close();

	}
	
	private void generateFrameMTD(String datafile) throws IllegalArgumentException, IOException 
	{
		JSONObject mtd = new JSONObject();
		
		mtd.put("data_type", "frame");
		mtd.put("format", "csv");
		mtd.put("header", false);
		
		FileSystem fs = FileSystem.get(TestUtils.conf);
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(datafile+".mtd"),true)));
		out.write(mtd.toString());
		out.close();
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 */
	private void runScalingTest( int rows, int cols, ExecType et, String ofmt) throws IOException, DMLRuntimeException
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			String specFile = HOME + INPUT_DIR + "spec.json";
			String inputFile = HOME + INPUT_DIR + "X";
			String outputFile = HOME + OUTPUT_DIR + config.getOutputFiles()[0];
			String outputFileR = HOME + EXPECTED_DIR + config.getOutputFiles()[0];
			
			generateSpecFile(cols, specFile);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-nvargs", 
					"DATA=" + inputFile,
					"TFSPEC=" + specFile,
					"TFMTD=" + HOME + OUTPUT_DIR + "tfmtd",
					"TFDATA=" + outputFile,
					"OFMT=" + ofmt
              };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       inputFile + " " + outputFileR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(rows, cols, -50, 50, 1.0, 7); 
			TestUtils.writeCSVTestMatrix(inputFile, X);
			generateFrameMTD(inputFile);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			ReaderTextCSV expReader=  new ReaderTextCSV(new CSVFileFormatProperties(false, ",", true, 0, null));
			MatrixBlock exp = expReader.readMatrixFromHDFS(outputFileR, -1, -1, -1, -1, -1);
			MatrixBlock out = null;
			
			if ( ofmt.equals("csv") ) 
			{
				ReaderTextCSV outReader=  new ReaderTextCSV(new CSVFileFormatProperties(false, ",", true, 0, null));
				out = outReader.readMatrixFromHDFS(outputFile, -1, -1, -1, -1, -1);
			}
			else
			{
				ReaderBinaryBlock bbReader = new ReaderBinaryBlock(false);
				out = bbReader.readMatrixFromHDFS(
						outputFile, exp.getNumRows(), exp.getNumColumns(), 
						ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ), 
						ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ),
						-1);
			}
			
			assertTrue("Incorrect output from data transform.", TransformTest.equals(out,exp,  1e-10));
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}	
}