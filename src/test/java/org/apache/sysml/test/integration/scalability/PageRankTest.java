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

package org.apache.sysml.test.integration.scalability;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedScalabilityTestBase;
import org.apache.sysml.test.integration.TestConfiguration;



public class PageRankTest extends AutomatedScalabilityTestBase 
{
	private static final String TEST_DIR = "test/scripts/scalability/page_rank/";
	
	
    @Override
    public void setUp() {
        addTestConfiguration("PageRankTest", new TestConfiguration(TEST_DIR, "PageRankTest", new String[] { "p" }));
        matrixSizes = new int[][] {
                { 9914 }
        };
    }
    
    @Test
    public void testPageRank() {
    	TestConfiguration config = getTestConfiguration("PageRankTest");
    	loadTestConfiguration(config);
        
        addInputMatrix("g", -1, -1, 1, 1, 0.000374962, -1).setRowsIndexInMatrixSizes(0).setColsIndexInMatrixSizes(0);
        addInputMatrix("p", -1, 1, 1, 1, 1, -1).setRowsIndexInMatrixSizes(0);
        addInputMatrix("e", -1, 1, 1, 1, 1, -1).setRowsIndexInMatrixSizes(0);
        addInputMatrix("u", 1, -1, 1, 1, 1, -1).setColsIndexInMatrixSizes(0);
        
        runTest();
    }

}
