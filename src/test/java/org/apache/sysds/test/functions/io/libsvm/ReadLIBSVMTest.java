/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.io.libsvm;

import org.junit.Test;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.conf.CompilerConfig;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class ReadLIBSVMTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "ReadLIBSVMTest";
	private final static String TEST_DIR = "functions/io/libsvm/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReadLIBSVMTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) );  
	}
	
	@Test
	public void testlibsvm1_Seq_CP() {
		runlibsvmTest(1, ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testlibsvm2_Seq_CP() {
		runlibsvmTest(2, ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testlibsvm2_Pllel_CP() {
		runlibsvmTest(2, ExecMode.SINGLE_NODE, true);
	}
	
	private void runlibsvmTest (int testNumber, ExecMode platform, boolean parallel)
	{
		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;
		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;
		
		try
		{
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrix = HOME + "test" + testNumber+".libsvm";
			String dmlOutput = output("dml.scalar");
			
			fullDMLScriptName = HOME + TEST_NAME + "_" + testNumber + ".dml";
			programArgs = new String[]{"-explain", "hops", "-args", inputMatrix, dmlOutput};
			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
		}
	}
}
