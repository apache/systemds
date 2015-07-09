/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.transform;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.runtime.io.ReaderBinaryBlock;
import com.ibm.bi.dml.runtime.io.ReaderTextCSV;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class TransformTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "transform";
	private final static String TEST_NAME2 = "apply";
	private final static String TEST_DIR = "functions/transform/";
	
	private final static String HOMES_DATASET 	= "homes/homes.csv";
	private final static String HOMES_SPEC 		= "homes/homes.tfspec.json";
	private final static String HOMES_TFDATA 	= "homes/homes.transformed.csv";
	
	private final static String IRIS_DATASET 	= "iris/iris.csv";
	private final static String IRIS_SPEC 		= "iris/iris.tfspec.json";
	private final static String IRIS_TFDATA 	= "iris/iris.transformed.csv";
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "y" })   ); 
	}
	
	@Test
	public void testIrisHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "iris");
	}
	
	@Test
	public void testIrisHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "iris");
	}
	
	@Test
	public void testIrisHadoopCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "iris");
	}

	@Test
	public void testIrisHadoopBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "iris");
	}
	
	@Test
	public void testHomesHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "homes");
	}
	
	@Test
	public void testHomesHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "homes");
	}
	
	@Test
	public void testHomesHadoopCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "homes");
	}

	@Test
	public void testHomesHadoopBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "homes");
	}
	
	private void runTransformTest( RUNTIME_PLATFORM rt, String ofmt, String dataset )
	{
		String DATASET = null, SPEC=null, TFDATA=null;
		
		if(dataset.equals("homes"))
		{
			DATASET = HOMES_DATASET;
			SPEC = HOMES_SPEC;
			TFDATA = HOMES_TFDATA;
		}
		else if (dataset.equals("iris"))
		{
			DATASET = IRIS_DATASET;
			SPEC = IRIS_SPEC;
			TFDATA = IRIS_TFDATA;
		}

		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-nvargs", 
											"DATA=" + HOME + "input/" + DATASET,
											"TFSPEC=" + HOME + "input/" + SPEC,
											"TFMTD=" + HOME + OUTPUT_DIR + "tfmtd",
											"TFDATA=" + HOME + OUTPUT_DIR + "tfout",
											"OFMT=" + ofmt
					                  };
			
			loadTestConfiguration(config);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			fullDMLScriptName = HOME + TEST_NAME2 + ".dml";
			programArgs = new String[]{"-nvargs", 
											"DATA=" + HOME + "input/" + DATASET,
											"APPLYMTD=" + HOME + OUTPUT_DIR + "tfmtd",  // generated above
											"TFMTD=" + HOME + OUTPUT_DIR + "test_tfmtd",
											"TFDATA=" + HOME + OUTPUT_DIR + "test_tfout",
											"OFMT=" + ofmt
					                  };
			
			loadTestConfiguration(config);
	
			exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			
			
			try {
				ReaderTextCSV csvReader=  new ReaderTextCSV(new CSVFileFormatProperties(true, ",", true, 0, null));
				MatrixBlock exp = csvReader.readMatrixFromHDFS(HOME+"input/"+ TFDATA, -1, -1, -1, -1, -1);
				
				MatrixBlock out = null, out2=null;
				if(ofmt.equals("csv"))
				{
					ReaderTextCSV outReader=  new ReaderTextCSV(new CSVFileFormatProperties(false, ",", true, 0, null));
					out = outReader.readMatrixFromHDFS(HOME+OUTPUT_DIR+"tfout", -1, -1, -1, -1, -1);
					out2 = outReader.readMatrixFromHDFS(HOME+OUTPUT_DIR+"test_tfout", -1, -1, -1, -1, -1);
				}
				else
				{
					ReaderBinaryBlock bbReader = new ReaderBinaryBlock(false);
					out = bbReader.readMatrixFromHDFS(
							HOME+OUTPUT_DIR+"tfout", exp.getNumRows(), exp.getNumColumns(), 
							ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ), 
							ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ),
							-1);
					out2 = bbReader.readMatrixFromHDFS(
							HOME+OUTPUT_DIR+"test_tfout", exp.getNumRows(), exp.getNumColumns(), 
							ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ), 
							ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ),
							-1);
				}
				
				assertTrue("Incorrect output from data transform.", equals(out,exp));
				assertTrue("Incorrect output from apply transform.", equals(out2,exp));
					
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}
		finally
		{
			rtplatform = rtold;
		}
	}
	
	private static boolean equals(MatrixBlock mb1, MatrixBlock mb2)
	{
		if(mb1.getNumRows() != mb2.getNumRows() || mb1.getNumColumns() != mb2.getNumColumns() || mb1.getNonZeros() != mb2.getNonZeros() )
			return false;
		
		// TODO: this implementation is to be optimized for different block representations
		for(int i=0; i < mb1.getNumRows(); i++) 
			for(int j=0; j < mb1.getNumColumns(); j++ )
				if(mb1.getValue(i, j) != mb2.getValue(i,j)) 
					return false;
		
		return true;
	}
	
}