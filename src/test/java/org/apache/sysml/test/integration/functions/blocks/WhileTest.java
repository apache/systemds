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

package org.apache.sysml.test.integration.functions.blocks;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;


/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>inner loop computation</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class WhileTest extends AutomatedTestBase 
{

	private static final String TEST_DIR = "functions/blocks/";
	private static String TEST_CLASS_DIR = TEST_DIR + WhileTest.class.getSimpleName() + "/";

    @Override
    public void setUp() {
        // positive tests
        addTestConfiguration("ComputationTest", new TestConfiguration(TEST_CLASS_DIR, "WhileTest",
        		new String[] { "b" }));
        addTestConfiguration("CleanUpTest", new TestConfiguration(TEST_CLASS_DIR, "WhileTest",
                new String[] { "b" }));

        // negative tests
    }
    
    @Test
    public void testComputation() {
    	int rows = 10;
    	int cols = 10;
        int maxIterations = 3;
        
    	TestConfiguration config = availableTestConfigurations.get("ComputationTest");
    	config.addVariable("rows", rows);
    	config.addVariable("cols", cols);
    	config.addVariable("maxiterations", maxIterations);
    	
    	loadTestConfiguration(config);
        
        double[][] a = getRandomMatrix(rows, cols, -1, 1, 1, -1);
        double[][] b = new double[rows][cols];
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                b[i][j] = Math.pow(3, maxIterations) * a[i][j];
            }
        }
        
        writeInputMatrix("a", a);
        writeExpectedMatrix("b", b);
        
        runTest();
        
        compareResults(1e-14);
    }

    @Test
    public void testCleanUp() {
        int rows = 10;
        int cols = 10;
        int maxIterations = 3;

        TestConfiguration config = availableTestConfigurations.get("CleanUpTest");
        config.addVariable("rows", rows);
        config.addVariable("cols", cols);
        config.addVariable("maxiterations", maxIterations);

        loadTestConfiguration(config);

        createRandomMatrix("a", rows, cols, -1, 1, 1, -1);

        runTest();

        Assert.assertFalse("This process's temp directory was not removed",
                checkForProcessLocalTemporaryDir());
    }
}
