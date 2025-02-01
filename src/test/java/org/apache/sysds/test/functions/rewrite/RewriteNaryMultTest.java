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

package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.BuiltinNaryCPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class RewriteNaryMultTest extends AutomatedTestBase
{
	private static final String TEST_NAME1 = "RewriteNaryMultDense1";
	private static final String TEST_NAME2 = "RewriteNaryMultDense2";
	private static final String TEST_NAME3 = "RewriteNaryMultSparse1";

	private static final String TEST_NAME4 = "RewriteNaryMultSparse2";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteNaryMultTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}) );
	}

	@Test
	public void testExpectExceptionParseNaryCPInstruction(){
		String opcode = "nxy";
		try {
			BuiltinNaryCPInstruction.parseInstruction("CP°" + opcode +"°A·MATRIX·FP64°B·MATRIX·FP64°C·MATRIX·FP64°_mVar3·MATRIX·FP64");
		}
		catch (DMLRuntimeException e){
			assert e.getMessage().equals("Opcode (" + opcode + ") not recognized in BuiltinMultipleCPInstruction");
		}
	}

	@Test
	public void testNoRewriteDense1CP() {
		testRewriteNaryMult(TEST_NAME1, false, ExecType.CP);
	}

	@Test
	public void testRewriteDense1CP() {
		testRewriteNaryMult(TEST_NAME1, true, ExecType.CP);
	}

	@Test
	public void testRewriteDense2CP() {
		testRewriteNaryMult(TEST_NAME2, true, ExecType.CP);
	}

	@Test
	public void testRewriteSparse1CP() {
		testRewriteNaryMult(TEST_NAME3, true, ExecType.CP);
	}

	@Test
	public void testRewriteSparse2CP() {
		testRewriteNaryMult(TEST_NAME4, true, ExecType.CP);
	}

	@Test
	public void testNoRewriteDense1SP() {
		testRewriteNaryMult(TEST_NAME1, false, ExecType.SPARK);
	}

	@Test
	public void testRewriteDense1SP() {
		testRewriteNaryMult(TEST_NAME1, true, ExecType.SPARK);
	}

	@Test
	public void testRewriteDense2SP() {
		testRewriteNaryMult(TEST_NAME2, true, ExecType.SPARK);
	}

	@Test
	public void testRewriteSparse1SP() {
		testRewriteNaryMult(TEST_NAME3, true, ExecType.SPARK);
	}

	@Test
	public void testRewriteSparse2SP() {
		testRewriteNaryMult(TEST_NAME4, true, ExecType.SPARK);
	}


	private void testRewriteNaryMult(String name, boolean rewrites, ExecType et)
	{
		ExecMode oldMode = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(name);
			loadTestConfiguration(config);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + name + ".dml";
			programArgs = new String[]{"-explain","-stats","-args", output("R") };

			runTest(true, false, null, -1); 
			
			//compare output
			Double ret = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
			if(name.equals(TEST_NAME3))
				Assert.assertEquals(Double.valueOf(1 + 304 + 10000), ret);
			else if(name.equals(TEST_NAME4))
				Assert.assertEquals(Double.valueOf(1), ret, 1e-7);
			else
				Assert.assertEquals(Double.valueOf(2*3*4*5*6*1000), ret);
			
			//check for applied nary plus
			String prefix = et == ExecType.SPARK ? "sp_" : "";
			if( rewrites && !name.equals(TEST_NAME2) )
				Assert.assertEquals(1, Statistics.getCPHeavyHitterCount(prefix + Opcodes.NM.toString()));
			else
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(prefix+"*")>=1);
		}
		finally {
			resetExecMode(oldMode);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
