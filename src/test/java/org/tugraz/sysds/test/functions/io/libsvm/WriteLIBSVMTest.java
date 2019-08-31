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

import java.io.IOException;

import org.junit.Test;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;

public class WriteLIBSVMTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "WriteLIBSVMTest";
	private final static String TEST_DIR = "functions/io/libsvm/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WriteLIBSVMTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
		new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) );  
	}
	
	@Test
	public void testLIBSVM1_CP() throws IOException {
		runWriteLIBSVMTest(ExecMode.SINGLE_NODE, 1, 1, false);
	}

	@Test
	public void testLIBSVM2_CP() throws IOException {
		runWriteLIBSVMTest(ExecMode.SINGLE_NODE, 1, 2, true);
	}

	@Test
	public void testLIBSVM3_CP() throws IOException {
		runWriteLIBSVMTest(ExecMode.SINGLE_NODE, 2, 3, true);
	}
	
	private void runWriteLIBSVMTest(ExecMode platform, int tno, int mno, boolean sparse) throws IOException {
		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		String inputMatrixName = HOME + "test" + mno +"w.libsvm";
		String dmlOutput = output("dml.scalar");
		String libsvmOutputName = output("libsvm_dml.data");
		
		fullDMLScriptName = HOME + TEST_NAME + "_" + tno + ".dml";
		programArgs = new String[]{"-explain", "-args", inputMatrixName, dmlOutput, libsvmOutputName,
			Boolean.toString(sparse) };
		
		runTest(true, false, null, -1);
		rtplatform = oldPlatform;
	}
}
