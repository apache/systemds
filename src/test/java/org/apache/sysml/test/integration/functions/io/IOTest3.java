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

package org.apache.sysml.test.integration.functions.io;

import org.junit.Test;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;


/**
 * <p>
 * <b>Positive tests:</b>
 * </p>
 * <ul>
 * <li>text format</li>
 * <li>binary format</li>
 * </ul>
 * <p>
 * <b>Negative tests:</b>
 * </p>
 * <ul>
 * </ul>
 * 
 * 
 */
public class IOTest3 extends AutomatedTestBase 
{
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/io/";

		// positive tests
		
		

		// negative tests
		availableTestConfigurations.put("SimpleTest", new TestConfiguration("functions/io/", "IOTest3", new String[] { "a" }));
	}

	@Test
	public void testSimple() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = availableTestConfigurations.get("SimpleTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
	
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a);
		writeExpectedMatrix("a", a);

		runTest(true, DMLException.class);

	}

}
