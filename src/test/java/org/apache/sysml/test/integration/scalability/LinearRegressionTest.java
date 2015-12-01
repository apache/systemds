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

package org.apache.sysml.test.integration.scalability;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedScalabilityTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;



public class LinearRegressionTest extends AutomatedScalabilityTestBase 
{
	
	private static final String TEST_DIR = "test/scripts/scalability/linear_regression/";
	
    @Override
    public void setUp() {
    	TestUtils.clearAssertionInformation();
        addTestConfiguration("LinearRegressionTest", new TestConfiguration(TEST_DIR, "LinearRegressionTest", new String[] { "w" }));
        matrixSizes = new int[][] {
                { 19004, 15436 }
        };
    }
    
    @Test
    public void testLinearRegression() {
    	TestConfiguration config = getTestConfiguration("LinearRegressionTest");
    	loadTestConfiguration(config);
        
        addInputMatrix("g", -1, -1, 0, 1, 0.00594116, -1).setRowsIndexInMatrixSizes(0).setColsIndexInMatrixSizes(1);
        addInputMatrix("b", -1, 1, 1, 10, 1, -1).setRowsIndexInMatrixSizes(0);
        
        runTest();
    }

}
