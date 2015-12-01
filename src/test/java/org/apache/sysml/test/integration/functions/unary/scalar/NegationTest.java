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

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>negation (int, negative int, double, negative double)</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class NegationTest extends AutomatedTestBase 
{
	
	private static String TEST_DIR = "functions/unary/scalar/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		
		// positive tests
		addTestConfiguration("NegationTest", new TestConfiguration(TEST_DIR, "NegationTest", new String[] { }));
		
		// negative tests
	}
	
	@Test
	public void testNegation() {
		int intValue = 2;
		int negativeIntValue = -2;
		double doubleValue = 2;
		double negativeDoubleValue = -2;
		
		TestConfiguration config = availableTestConfigurations.get("NegationTest");
		config.addVariable("int", intValue);
		config.addVariable("negativeint", negativeIntValue);
		config.addVariable("double", doubleValue);
		config.addVariable("negativedouble", negativeDoubleValue);
		
		loadTestConfiguration(config);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("int", -intValue);
		writeExpectedHelperMatrix("negative_int", -negativeIntValue);
		writeExpectedHelperMatrix("double", -doubleValue);
		writeExpectedHelperMatrix("negative_double", -negativeDoubleValue);
		
		runTest();
		
		compareResults();
	}
	
}
