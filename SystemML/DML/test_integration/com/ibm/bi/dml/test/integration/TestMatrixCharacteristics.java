/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration;

public class TestMatrixCharacteristics 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    private String matrixName;
    private int rows;
    private int rowsIndexInMatrixSizes;
    private int cols;
    private int colsIndexInMatrixSizes;
    private double minValue;
    private double maxValue;
    private double sparsity;
    private long seed;
    
    
    public TestMatrixCharacteristics(String matrixName, int rows, int cols, double minValue, double maxValue,
            double sparsity, long seed) {
        this.matrixName = matrixName;
        this.rows = rows;
        this.cols = cols;
        this.minValue = minValue;
        this.maxValue = maxValue;
        this.sparsity = sparsity;
        this.seed = seed;
    }

    public String getMatrixName() {
        return matrixName;
    }

    public int getRows() {
        return rows;
    }
    
    public void setRows(int rows) {
        this.rows = rows;
    }
    
    public int getRowsIndexInMatrixSizes() {
        return rowsIndexInMatrixSizes;
    }
    
    public TestMatrixCharacteristics setRowsIndexInMatrixSizes(int rowsIndexInMatrixSizes) {
        this.rowsIndexInMatrixSizes = rowsIndexInMatrixSizes;
        return this;
    }

    public int getCols() {
        return cols;
    }
    
    public void setCols(int cols) {
        this.cols = cols;
    }
    
    public int getColsIndexInMatrixSizes() {
        return colsIndexInMatrixSizes;
    }
    
    public TestMatrixCharacteristics setColsIndexInMatrixSizes(int colsIndexInMatrixSizes) {
        this.colsIndexInMatrixSizes = colsIndexInMatrixSizes;
        return this;
    }

    public double getMinValue() {
        return minValue;
    }

    public double getMaxValue() {
        return maxValue;
    }

    public double getSparsity() {
        return sparsity;
    }

    public long getSeed() {
        return seed;
    }
    
}
