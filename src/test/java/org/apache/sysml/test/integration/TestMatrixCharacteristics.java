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

package com.ibm.bi.dml.test.integration;

public class TestMatrixCharacteristics 
{

	
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
