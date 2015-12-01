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

package org.apache.sysml.test.integration;

import java.util.ArrayList;

import org.junit.After;
import org.junit.Before;

import org.apache.sysml.test.utils.TestUtils;


public abstract class AutomatedScalabilityTestBase extends AutomatedTestBase 
{

	    
    private static final boolean RUN_SCALABILITY_TESTS = false;

    private long[] timeMeasurements;
    protected int[][] matrixSizes;
    protected ArrayList<TestMatrixCharacteristics> inputMatrices;
    
    
    @Before
    public void setUpScalabilityTest() {
        inputMatrices = new ArrayList<TestMatrixCharacteristics>();
    }
    
    public abstract void setUp();
    
    protected void runTest() {
        if(!RUN_SCALABILITY_TESTS)
            return;
        
        timeMeasurements = new long[matrixSizes.length];
        for(int i = 0; i < matrixSizes.length; i++) {
            for(TestMatrixCharacteristics inputMatrix : inputMatrices) {
                if(inputMatrix.getRows() == -1)
                    inputMatrix.setRows(matrixSizes[i][inputMatrix.getRowsIndexInMatrixSizes()]);
                if(inputMatrix.getCols() == -1)
                    inputMatrix.setCols(matrixSizes[i][inputMatrix.getColsIndexInMatrixSizes()]);
                createRandomMatrix(inputMatrix);
            }
            
            for(int j = 0; j < matrixSizes[i].length; j++) {
            	testVariables.put(Integer.toString(j), Integer.toString(matrixSizes[i][j]));
            }

            long startingTime = System.currentTimeMillis();
            super.runTest();
            long finishingTime = System.currentTimeMillis();
            timeMeasurements[i] = (finishingTime - startingTime);
            
            TestUtils.renameTempDMLScript(baseDirectory + selectedTest + ".dml");
        }
    }
    
    protected TestMatrixCharacteristics addInputMatrix(String name, int rows, int cols, double min, double max,
            double sparsity, long seed) {
        TestMatrixCharacteristics inputMatrix = new TestMatrixCharacteristics(name, rows, cols, min, max,
                sparsity, seed);
        inputMatrices.add(inputMatrix);
        return inputMatrix;
    }
    
    @After
    public void displayTimeMeasurements() {
        if(!RUN_SCALABILITY_TESTS)
            return;
        
        for(long timeMeasurement : timeMeasurements) {
            System.out.println("measured time: " + timeMeasurement);
        }
    }

}
