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

package org.apache.sysds.test.functions.recompile;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.utils.Statistics;

public class LiteralReplaceCastScalarReadTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "LiteralReplaceCastScalar";
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + 
		LiteralReplaceCastScalarReadTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }));
	}

	
	@Test
	public void testRemoveCastsInputInteger() {
		runScalarCastTest(ValueType.INT64);
	}
	
	@Test
	public void testRemoveCastsInputDouble() {
		runScalarCastTest(ValueType.FP64);
	}
	
	@Test
	public void testRemoveCastsInputBoolean() {
		runScalarCastTest(ValueType.BOOLEAN);
	}
	
	private void runScalarCastTest( ValueType vt ) {
		boolean oldCF = OptimizerUtils.ALLOW_CONSTANT_FOLDING;
		
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			OptimizerUtils.ALLOW_CONSTANT_FOLDING = false;
			
			// input value
			String val = null;
			switch( vt ) {
				case INT64: val = "7"; break;
				case FP64: val = "7.3"; break;
				case BOOLEAN: val = "TRUE"; break;
				default: //do nothing
			}
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			//note: stats required for runtime check of rewrite
			programArgs = new String[]{"-stats","-args", val };
			
			runTest(true, false, null, -1); 
		
			//CHECK cast replacement and sum replacement
			Assert.assertEquals(false, Statistics.getCPHeavyHitterOpCodes().contains(Opcodes.CAST_AS_INT.toString()));
			Assert.assertEquals(false, Statistics.getCPHeavyHitterOpCodes().contains(Opcodes.CAST_AS_DOUBLE.toString()));
			Assert.assertEquals(false, Statistics.getCPHeavyHitterOpCodes().contains(Opcodes.CAST_AS_BOOLEAN.toString()));
			Assert.assertEquals(false, Statistics.getCPHeavyHitterOpCodes().contains(Opcodes.UAKP.toString())); //sum
		}
		finally {
			OptimizerUtils.ALLOW_CONSTANT_FOLDING = oldCF;
		}
	}
}
