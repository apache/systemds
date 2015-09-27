/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

/**
 *   
 */
public class SetWorkingDirTest extends AutomatedTestBase
{	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "PackageFunCall1";
	private final static String TEST_NAME2 = "PackageFunCall2";
	private final static String TEST_NAME0 = "PackageFunLib";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
	}
	
	@Test
	public void testDefaultWorkingDirDml() { 
		runTest( TEST_NAME1, false, ScriptType.DML ); 
	}
	
	@Test
	public void testSetWorkingDirDml() { 
		runTest( TEST_NAME2, false, ScriptType.DML ); 
	}
	
	@Test
	public void testDefaultWorkingDirFailDml() { 
		runTest( TEST_NAME1, true, ScriptType.DML ); 
	}
	
	@Test
	public void testSetWorkingDirFailDml() { 
		runTest( TEST_NAME2, true, ScriptType.DML ); 
	}

	@Test
	public void testDefaultWorkingDirPyDml() { 
		runTest( TEST_NAME1, false, ScriptType.PYDML ); 
	}
	
	@Test
	public void testSetWorkingDirPyDml() { 
		runTest( TEST_NAME2, false, ScriptType.PYDML ); 
	}
	
	@Test
	public void testDefaultWorkingDirFailPyDml() { 
		runTest( TEST_NAME1, true, ScriptType.PYDML ); 
	}
	
	@Test
	public void testSetWorkingDirFailPyDml() { 
		runTest( TEST_NAME2, true, ScriptType.PYDML ); 
	}
	
	/**
	 * 
	 * @param testName
	 * @param exceptionExpected
	 * @param scriptType
	 */
	private void runTest( String testName, boolean exceptionExpected, ScriptType scriptType ) 
	{
		
		//construct source filenames of dml scripts 
		String dir = SCRIPT_DIR + TEST_DIR;
		String nameCall = testName + "." + scriptType.lowerCase();
		String nameLib = TEST_NAME0 + "." + scriptType.lowerCase();
		
		try 
		{
			//copy dml/pydml scripts to current dir
			FileUtils.copyFile(new File(dir+nameCall), new File(nameCall));
			if( !exceptionExpected )
				FileUtils.copyFile(new File(dir+nameLib), new File(nameLib));
			
			//setup test configuration
			TestConfiguration config = getTestConfiguration(testName);
		    fullDMLScriptName = nameCall;
		    if (scriptType == ScriptType.PYDML) {
		    	programArgs = new String[]{ "-python" };
		    } else {
		    	programArgs = new String[]{};
		    }
		    loadTestConfiguration(config);
			
			//run tests
	        runTest(true, exceptionExpected, DMLException.class, -1);
		} 
		catch (IOException e) {
			throw new RuntimeException(e);
		}
		finally 
		{
			//delete dml/pydml scripts from current dir (see above)
			LocalFileUtils.deleteFileIfExists(nameCall);
			if( !exceptionExpected )
				LocalFileUtils.deleteFileIfExists(nameLib);
		}
	}
}
