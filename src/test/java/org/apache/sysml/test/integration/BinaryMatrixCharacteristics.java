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

package org.apache.sysml.test.integration;


/**
 * <p>Contains characteristics about a binary matrix.</p>
 * 
 * 
 */
public class BinaryMatrixCharacteristics 
{

	    
    private double[][] values;
    private int rows;
    private int cols;
    private int rowsInBlock;
    private int rowsInLastBlock;
    private int colsInBlock;
    private int colsInLastBlock;
    private long nonZeros;
    
    
    public BinaryMatrixCharacteristics(double[][] values, int rows, int cols, int rowsInBlock, int rowsInLastBlock,
            int colsInBlock, int colsInLastBlock, long nonZeros) {
        this.values = values;
        this.rows = rows;
        this.cols = cols;
        this.rowsInBlock = rowsInBlock;
        this.rowsInLastBlock = rowsInLastBlock;
        this.colsInBlock = colsInBlock;
        this.colsInLastBlock = colsInLastBlock;
        this.nonZeros = nonZeros;
    }
    
    public double[][] getValues() {
        return values;
    }
    
    public int getRows() {
        return rows;
    }
    
    public int getCols() {
        return cols;
    }
    
    public int getRowsInBlock() {
        return rowsInBlock;
    }
    
    public int getRowsInLastBlock() {
        return rowsInLastBlock;
    }
    
    public int getColsInBlock() {
        return colsInBlock;
    }
    
    public int getColsInLastBlock() {
        return colsInLastBlock;
    }
    
    public long getNonZeros() {
    	return nonZeros;
    }
    
}
