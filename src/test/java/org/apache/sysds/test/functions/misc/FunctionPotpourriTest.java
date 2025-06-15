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

package org.apache.sysds.test.functions.misc;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class FunctionPotpourriTest extends AutomatedTestBase 
{
	private final static String[] TEST_NAMES = new String[] {
		"FunPotpourriNoReturn",
		"FunPotpourriComments",
		"FunPotpourriNoReturn2",
		"FunPotpourriEval",
		"FunPotpourriSubsetReturn",
		"FunPotpourriSubsetReturnDead",
		"FunPotpourriNamedArgsSingle",
		"FunPotpourriNamedArgsMulti",
		"FunPotpourriNamedArgsPartial",
		"FunPotpourriNamedArgsUnknown1",
		"FunPotpourriNamedArgsUnknown2",
		"FunPotpourriNamedArgsIPA",
		"FunPotpourriDefaultArgScalar",
		"FunPotpourriDefaultArgMatrix",
		"FunPotpourriDefaultArgScalarMatrix1",
		"FunPotpourriDefaultArgScalarMatrix2",
		"FunPotpourriNamedArgsQuotedAssign",
		"FunPotpourriMultiReturnBuiltin1",
		"FunPotpourriMultiReturnBuiltin2",
		"FunPotpourriNestedParforEval",
		"FunPotpourriMultiEval",
		"FunPotpourriEvalPred",
		"FunPotpourriEvalList1Arg",
		"FunPotpourriEvalList2Arg",
		"FunPotpourriEvalNamespace",
		"FunPotpourriEvalNamespace2",
		"FunPotpourriBuiltinPrecedence",
		"FunPotpourriParforEvalBuiltin",
		"FunPotpourriParforEvalSpark",
		"FunPotpourriEvalNamespace3",
		"FunPotpourriDefaultParams",
		"FunPotpourriListHandling"
	};
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FunctionPotpourriTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(String testName : TEST_NAMES)
			addTestConfiguration( testName, new TestConfiguration(TEST_CLASS_DIR, testName, new String[] { "R" }) );
	}

	@Test
	public void testFunctionNoReturn() {
		runFunctionTest( TEST_NAMES[0], null );
	}
	
	@Test
	public void testFunctionComments() {
		runFunctionTest( TEST_NAMES[1], null );
	}
	
	@Test
	public void testFunctionNoReturnSpec() {
		runFunctionTest( TEST_NAMES[2], null );
	}
	
	@Test
	public void testFunctionEval() {
		runFunctionTest( TEST_NAMES[3], null );
	}
	
	@Test
	public void testFunctionEval2() {
		runFunctionTest( TEST_NAMES[3], null, true );
	}
	
	@Test
	public void testFunctionSubsetReturn() {
		runFunctionTest( TEST_NAMES[4], null );
	}
	
	@Test
	public void testFunctionSubsetReturnDead() {
		runFunctionTest( TEST_NAMES[5], null );
	}
	
	@Test
	public void testFunctionNamedArgsSingleErr() {
		runFunctionTest( TEST_NAMES[6], ParseException.class );
	}
	
	@Test
	public void testFunctionNamedArgsMultiErr() {
		runFunctionTest( TEST_NAMES[7], ParseException.class );
	}

	@Test
	public void testFunctionNamedArgsPartial() {
		runFunctionTest( TEST_NAMES[8], LanguageException.class );
	}
	
	@Test
	public void testFunctionNamedArgsUnkown1() {
		runFunctionTest( TEST_NAMES[9], LanguageException.class );
	}
	
	@Test
	public void testFunctionNamedArgsUnkown2() {
		runFunctionTest( TEST_NAMES[10], LanguageException.class );
	}
	
	@Test
	public void testFunctionNamedArgsIPA() {
		runFunctionTest( TEST_NAMES[11], null );
	}
	
	@Test
	public void testFunctionDefaultArgsScalar() {
		runFunctionTest( TEST_NAMES[12], null );
	}
	
	@Test
	public void testFunctionDefaultArgsMatrix() {
		runFunctionTest( TEST_NAMES[13], null );
	}
	
	@Test
	public void testFunctionDefaultArgsScalarMatrix1() {
		runFunctionTest( TEST_NAMES[14], null );
	}
	
	@Test
	public void testFunctionDefaultArgsScalarMatrix2() {
		runFunctionTest( TEST_NAMES[15], null );
	}
	
	@Test
	public void testFunctionNamedArgsQuotedAssign() {
		runFunctionTest( TEST_NAMES[16], null );
	}
	
	@Test
	public void testFunctionMultiReturnBuiltin1() {
		runFunctionTest( TEST_NAMES[17], null );
	}
	
	@Test
	public void testFunctionMultiReturnBuiltin2() {
		runFunctionTest( TEST_NAMES[18], null );
	}
	
	@Test
	public void testFunctionNestedParforEval() {
		runFunctionTest( TEST_NAMES[19], null );
	}
	
	@Test
	@Ignore //TODO support list
	public void testFunctionNestedParforEval2() {
		runFunctionTest( TEST_NAMES[19], null, true );
	}
	
	@Test
	public void testFunctionMultiEval() {
		runFunctionTest( TEST_NAMES[20], null );
	}
	
	@Test
	@Ignore //TODO support list
	public void testFunctionMultiEval2() {
		runFunctionTest( TEST_NAMES[20], null, true );
	}
	
	@Test
	public void testFunctionEvalPred() {
		runFunctionTest( TEST_NAMES[21], null );
	}
	
	@Test
	@Ignore //TODO support list
	public void testFunctionEvalPred2() {
		runFunctionTest( TEST_NAMES[21], null, true );
	}
	
	@Test
	public void testFunctionEvalList1Arg() {
		runFunctionTest( TEST_NAMES[22], null );
	}
	
	@Test
	@Ignore //TODO support list
	public void testFunctionEvalList1Arg2() {
		runFunctionTest( TEST_NAMES[22], null, true );
	}
	
	@Test
	public void testFunctionEvalList2Arg() {
		runFunctionTest( TEST_NAMES[23], null );
	}
	
	@Test
	@Ignore //TODO support list
	public void testFunctionEvalList2Arg2() {
		runFunctionTest( TEST_NAMES[23], null, true );
	}
	
	@Test
	public void testFunctionEvalNamespace() {
		runFunctionTest( TEST_NAMES[24], null );
	}
	
	@Test
	@Ignore //TODO support list
	public void testFunctionEvalNamespace2() {
		runFunctionTest( TEST_NAMES[24], null, true );
	}
	
	@Test
	public void testFunctionEvalNamespacePlain() {
		runFunctionTest( TEST_NAMES[25], null );
	}
	
	@Test
	public void testFunctionEvalNamespacePlain2() {
		runFunctionTest( TEST_NAMES[25], null, true );
	}
	
	@Test
	public void testFunctionBuiltinPrecedence() {
		runFunctionTest( TEST_NAMES[26], null );
	}
	
	@Test
	public void testFunctionParforEvalBuiltin() {
		runFunctionTest( TEST_NAMES[27], null );
	}
	
	@Test
	@Ignore //TODO support list
	public void testFunctionParforEvalBuiltin2() {
		runFunctionTest( TEST_NAMES[27], null, true );
	}
	
	@Test
	public void testFunctionParforEvalSpark() {
		runFunctionTest( TEST_NAMES[28], null, true );
	}
	
	@Test
	public void testFunctionEvalNamespace3() {
		runFunctionTest( TEST_NAMES[29], null, false );
	}
	
	@Test
	public void testFunctionDefaultParams() {
		runFunctionTest( TEST_NAMES[30], null, false );
	}
	
	@Test
	public void testFunctionListHandling() {
		runFunctionTest( TEST_NAMES[31], null, false );
	}
	
	private void runFunctionTest(String testName, Class<?> error) {
		runFunctionTest(testName, error, false);
	}
	
	private void runFunctionTest(String testName, Class<?> error, boolean evalRewrite) {
		TestConfiguration config = getTestConfiguration(testName);
		loadTestConfiguration(config);
		
		boolean oldFlag = OptimizerUtils.ALLOW_EVAL_FCALL_REPLACEMENT;
		try {
			OptimizerUtils.ALLOW_EVAL_FCALL_REPLACEMENT = evalRewrite;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[]{"-explain", "hops", "-stats",
				"-args", String.valueOf(error).toUpperCase()};
	
			runTest(true, error != null, error, -1);
	
			if( testName.equals(TEST_NAMES[17]) )
				Assert.assertTrue(heavyHittersContainsString(Opcodes.PRINT.toString()));
			if( evalRewrite && !testName.equals(TEST_NAMES[28]) )
				Assert.assertTrue(!heavyHittersContainsString(Opcodes.EVAL.toString()));
			if( testName.equals(TEST_NAMES[31]) ) //print used for error
				Assert.assertFalse(heavyHittersContainsString(Opcodes.PRINT.toString()));
		}
		finally {
			OptimizerUtils.ALLOW_EVAL_FCALL_REPLACEMENT = oldFlag;
		}
	}
}
