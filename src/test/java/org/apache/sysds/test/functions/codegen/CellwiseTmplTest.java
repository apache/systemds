/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
 
package org.apache.sysds.test.functions.codegen;

import java.io.File;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class CellwiseTmplTest extends AutomatedTestBase 
{
	private static final Log LOG = LogFactory.getLog(CellwiseTmplTest.class.getName());

	private static final String TEST_NAME = "cellwisetmpl";
	private static final String TEST_NAME1 = TEST_NAME+1;
	private static final String TEST_NAME2 = TEST_NAME+2;
	private static final String TEST_NAME3 = TEST_NAME+3;
	private static final String TEST_NAME4 = TEST_NAME+4;
	private static final String TEST_NAME5 = TEST_NAME+5;
	private static final String TEST_NAME6 = TEST_NAME+6;
	private static final String TEST_NAME7 = TEST_NAME+7;
	private static final String TEST_NAME8 = TEST_NAME+8;
	private static final String TEST_NAME9 = TEST_NAME+9;   //sum((X + 7 * Y)^2)
	private static final String TEST_NAME10 = TEST_NAME+10; //min/max(X + 7 * Y)
	private static final String TEST_NAME11 = TEST_NAME+11; //replace((0 / (X - 500))+1, 0/0, 7)
	private static final String TEST_NAME12 = TEST_NAME+12; //((X/3) %% 0.6) + ((X/3) %/% 0.6)
	private static final String TEST_NAME13 = TEST_NAME+13; //min(X + 7 * Y) large
	private static final String TEST_NAME14 = TEST_NAME+14; //-2 * X + t(Y); t(Y) is rowvector
	private static final String TEST_NAME15 = TEST_NAME+15; //colMins(2*log(X))
	private static final String TEST_NAME16 = TEST_NAME+16; //colSums(2*log(X));
	private static final String TEST_NAME17 = TEST_NAME+17; //xor operation
	private static final String TEST_NAME18 = TEST_NAME+18; //sum(ifelse(X,Y,Z))
	private static final String TEST_NAME19 = TEST_NAME+19; //sum(ifelse(true,Y,Z))+sum(ifelse(false,Y,Z))
	private static final String TEST_NAME20 = TEST_NAME+20; //bitwAnd() operation
	private static final String TEST_NAME21 = TEST_NAME+21; //relu operation, (X>0)*dout
	private static final String TEST_NAME22 = TEST_NAME+22; //sum(X * seq(1,N) + t(seq(M,1)))
	private static final String TEST_NAME23 = TEST_NAME+23; //sum(min(X,Y,Z))
	private static final String TEST_NAME24 = TEST_NAME+24; //min(X, Y, Z, 3, 7)
	private static final String TEST_NAME25 = TEST_NAME+25; //bias_add
	private static final String TEST_NAME26 = TEST_NAME+26; //bias_mult
	private static final String TEST_NAME27 = TEST_NAME+27; //outer < +7 negative
	private static final String TEST_NAME28 = TEST_NAME+28; //colProds(X^2 + 1)
	private static final String TEST_NAME29 = TEST_NAME+29; //colProds(2*log(X))
	private static final String TEST_NAME30 = TEST_NAME+30;	//rowProds(X^2 + 1)
	private static final String TEST_NAME31 = TEST_NAME+31;	//colProds(2*log(X))

	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CellwiseTmplTest.class.getSimpleName() + "/";
	private final static String TEST_CONF6 = "SystemDS-config-codegen6.xml";
	private final static String TEST_CONF7 = "SystemDS-config-codegen.xml";
	private static String TEST_CONF = TEST_CONF7;
	
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=31; i++ ) {
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(
				TEST_CLASS_DIR, TEST_NAME+i, new String[] {String.valueOf(i)}) );
		}
	}
	
	@Test
	public void testCodegenCellwiseRewrite1() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.CP );
	}
		
	@Test
	public void testCodegenCellwiseRewrite2() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite3() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite4() 
	{
		testCodegenIntegration( TEST_NAME4, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite5() {
		testCodegenIntegration( TEST_NAME5, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite6() {
		testCodegenIntegration( TEST_NAME6, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite7() {
		testCodegenIntegration( TEST_NAME7, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite8() {
		testCodegenIntegration( TEST_NAME8, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite9() {
		testCodegenIntegration( TEST_NAME9, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite10() {
		testCodegenIntegration( TEST_NAME10, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite11() {
		testCodegenIntegration( TEST_NAME11, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite12() {
		testCodegenIntegration( TEST_NAME12, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite13() {
		testCodegenIntegration( TEST_NAME13, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite14() {
		testCodegenIntegration( TEST_NAME14, true, ExecType.CP  );
	}

	@Test
	public void testCodegenCellwise1() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.CP );
	}
		
	@Test
	public void testCodegenCellwise2() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise3() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise4() { testCodegenIntegration( TEST_NAME4, false, ExecType.CP ); }
	
	@Test
	public void testCodegenCellwise5() { testCodegenIntegration( TEST_NAME5, false, ExecType.CP  );	}
	
	@Test
	public void testCodegenCellwise6() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise7() {
		testCodegenIntegration( TEST_NAME7, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise8() { testCodegenIntegration( TEST_NAME8, false, ExecType.CP  );	}
	
	@Test
	public void testCodegenCellwise9() {
		testCodegenIntegration( TEST_NAME9, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise10() {
		testCodegenIntegration( TEST_NAME10, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise11() {
		testCodegenIntegration( TEST_NAME11, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise12() {
		testCodegenIntegration( TEST_NAME12, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise13() {
		testCodegenIntegration( TEST_NAME13, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise14() {
		testCodegenIntegration( TEST_NAME14, false, ExecType.CP  );
	}

	@Test
	public void testCodegenCellwiseRewrite1_sp() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite7_sp() {
		testCodegenIntegration( TEST_NAME7, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite8_sp() {
		testCodegenIntegration( TEST_NAME8, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite9_sp() {
		testCodegenIntegration( TEST_NAME9, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite10_sp() {
		testCodegenIntegration( TEST_NAME10, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite11_sp() {
		testCodegenIntegration( TEST_NAME11, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite12_sp() {
		testCodegenIntegration( TEST_NAME12, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite13_sp() {
		testCodegenIntegration( TEST_NAME13, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite14_sp() {
		testCodegenIntegration( TEST_NAME14, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite15() {
		testCodegenIntegration( TEST_NAME15, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenCellwise15() {
		testCodegenIntegration( TEST_NAME15, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenCellwiseRewrite15_sp() {
		testCodegenIntegration( TEST_NAME15, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite16() {
		testCodegenIntegration( TEST_NAME16, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenCellwise16() {
		testCodegenIntegration( TEST_NAME16, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenCellwiseRewrite16_sp() {
		testCodegenIntegration( TEST_NAME16, true, ExecType.SPARK );
	}

	@Test
	public void testCodegenCellwiseRewrite17() {
		testCodegenIntegration( TEST_NAME17, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise17() {
		testCodegenIntegration( TEST_NAME17, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite17_sp() {
		testCodegenIntegration( TEST_NAME17, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite18() {
		testCodegenIntegration( TEST_NAME18, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise18() {
		testCodegenIntegration( TEST_NAME18, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite18_sp() {
		testCodegenIntegration( TEST_NAME18, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite19() {
		testCodegenIntegration( TEST_NAME19, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise19() {
		testCodegenIntegration( TEST_NAME19, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite19_sp() {
		testCodegenIntegration( TEST_NAME19, true, ExecType.SPARK );
	}

	@Test
	public void testCodegenCellwiseRewrite20() {
		testCodegenIntegration( TEST_NAME20, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise20() {
		testCodegenIntegration( TEST_NAME20, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite20_sp() {
		testCodegenIntegration( TEST_NAME20, true, ExecType.SPARK );
	}

	@Test
	public void testCodegenCellwiseRewrite21() {
		testCodegenIntegration( TEST_NAME21, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise21() {
		testCodegenIntegration( TEST_NAME21, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite21_sp() {
		testCodegenIntegration( TEST_NAME21, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite22() {
		testCodegenIntegration( TEST_NAME22, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise22() {
		testCodegenIntegration( TEST_NAME22, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite22_sp() {
		testCodegenIntegration( TEST_NAME22, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite23() {
		testCodegenIntegration( TEST_NAME23, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise23() {
		testCodegenIntegration( TEST_NAME23, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite23_sp() {
		testCodegenIntegration( TEST_NAME23, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite24() {
		testCodegenIntegration( TEST_NAME24, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise24() {
		testCodegenIntegration( TEST_NAME24, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite24_sp() {
		testCodegenIntegration( TEST_NAME24, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite25() {
		testCodegenIntegration( TEST_NAME25, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise25() {
		testCodegenIntegration( TEST_NAME25, false, ExecType.CP );
	}

	@Test //TODO handling of global col index
	public void testCodegenCellwiseRewrite25_sp() {
		testCodegenIntegration( TEST_NAME25, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite26() {
		testCodegenIntegration( TEST_NAME26, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise26() {
		testCodegenIntegration( TEST_NAME26, false, ExecType.CP );
	}

	@Test //TODO handling of global col index
	public void testCodegenCellwiseRewrite26_sp() {
		testCodegenIntegration( TEST_NAME26, true, ExecType.SPARK );
	}

	@Test
	public void testCodegenCellwiseRewrite27() {
		testCodegenIntegration( TEST_NAME27, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise27() {
		testCodegenIntegration( TEST_NAME27, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite27_sp() {
		testCodegenIntegration( TEST_NAME27, true, ExecType.SPARK );
	}

	@Test
	public void testCodegenCellwiseRewrite28() {
		testCodegenIntegration( TEST_NAME28, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise28() {
		testCodegenIntegration( TEST_NAME28, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite28_sp() {
		testCodegenIntegration( TEST_NAME28, true, ExecType.SPARK );
	}

	@Test
	public void testCodegenCellwiseRewrite29() {
		testCodegenIntegration( TEST_NAME29, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise29() {
		testCodegenIntegration( TEST_NAME29, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite29_sp() {
		testCodegenIntegration( TEST_NAME29, true, ExecType.SPARK );
	}

	@Test
	public void testCodegenCellwiseRewrite30() {
		testCodegenIntegration( TEST_NAME30, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise30() {
		testCodegenIntegration( TEST_NAME30, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite30_sp() {
		testCodegenIntegration( TEST_NAME30, true, ExecType.SPARK );
	}

	@Test
	public void testCodegenCellwiseRewrite31() {
		testCodegenIntegration( TEST_NAME31, true, ExecType.CP );
	}

	@Test
	public void testCodegenCellwise31() {
		testCodegenIntegration( TEST_NAME31, false, ExecType.CP );
	}

	@Test
	public void testCodegenCellwiseRewrite31_sp() {
		testCodegenIntegration( TEST_NAME31, true, ExecType.SPARK );
	}

	private void testCodegenIntegration( String testname, boolean rewrites, ExecType instType )
	{
		boolean oldRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		String oldTestConf = TEST_CONF;
		ExecMode platformOld = setExecMode(instType);
		
		if( testname.equals(TEST_NAME9) )
			TEST_CONF = TEST_CONF6;
		  
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "codegen", "-stats", "-args", output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1); 
			runRScript(true);

			if(testname.equals(TEST_NAME6) || testname.equals(TEST_NAME7) 
				|| testname.equals(TEST_NAME9) || testname.equals(TEST_NAME10)) {
				//compare scalars 
				HashMap<CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("S");
				HashMap<CellIndex, Double> rfile  = readRScalarFromExpectedDir("S");
				TestUtils.compareScalars((Double) dmlfile.values().toArray()[0], (Double) rfile.values().toArray()[0],0);
			}
			else {
				//compare matrices 
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("S");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}
			
			if( !(rewrites && (testname.equals(TEST_NAME2)
				|| testname.equals(TEST_NAME19))) && !testname.equals(TEST_NAME27) )
				Assert.assertTrue(heavyHittersContainsSubString(
						"spoofCell", "sp_spoofCell", "spoofMA", "sp_spoofMA", "gpu_spoofCUDACell"));
			if( testname.equals(TEST_NAME7) ) //ensure matrix mult is fused
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.TSMM.toString()));
			else if( testname.equals(TEST_NAME10) ) //ensure min/max is fused
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.UAMIN.toString(),Opcodes.UAMAX.toString()));
			else if( testname.equals(TEST_NAME11) ) //ensure replace is fused
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.REPLACE.toString()));
			else if( testname.equals(TEST_NAME15) )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.UACMIN.toString()));
			else if( testname.equals(TEST_NAME16) )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.UACKP.toString()));
			else if( testname.equals(TEST_NAME17) )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.XOR.toString()));
			else if( testname.equals(TEST_NAME22) )
				Assert.assertTrue(!heavyHittersContainsSubString("seq"));
			else if( testname.equals(TEST_NAME23) || testname.equals(TEST_NAME24) )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.MIN.toString(),Opcodes.NMIN.toString()));
		}
		finally {
			resetExecMode(platformOld);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewrites;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
			TEST_CONF = oldTestConf;
		}
	}

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
		LOG.debug("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
