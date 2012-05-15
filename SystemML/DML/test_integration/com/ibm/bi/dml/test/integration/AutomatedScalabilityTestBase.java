package com.ibm.bi.dml.test.integration;

import java.util.ArrayList;

import org.junit.After;
import org.junit.Before;

import com.ibm.bi.dml.test.utils.TestUtils;


public abstract class AutomatedScalabilityTestBase extends AutomatedTestBase {
    
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
