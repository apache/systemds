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

package org.apache.sysml.test.integration.functions.unary.scalar;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class RoundTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "RoundTest";
	private final static String TEST_DIR = "functions/unary/scalar/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("RoundTest", new TestConfiguration(TEST_DIR,"RoundTest", new String[] { "scalar" }));
	}
	
	@Test
	public void testRound() {
		TestConfiguration config = getTestConfiguration(TEST_NAME);

		double scalar = 10.7;
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", Double.toString(scalar), 
				                        HOME + OUTPUT_DIR + "scalar" };

		loadTestConfiguration(config);
		
		double roundScalar = Math.round(scalar);

		writeExpectedScalar("scalar", roundScalar);
		
		runTest(true, false, null, -1);
		
		HashMap<CellIndex, Double> map = readDMLScalarFromHDFS("scalar");
		double dmlvalue = map.get(new CellIndex(1,1));
		
		if ( dmlvalue != roundScalar ) {
			throw new RuntimeException("Values mismatch: DMLvalue " + dmlvalue + " != ExpectedValue " + roundScalar);
		}
		
		//compareResults();
	}
	
	
}
