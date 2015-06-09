/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration;


/**
 * <p>Contains characteristics about a binary matrix.</p>
 * 
 * 
 */
public class BinaryMatrixCharacteristics 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	    
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
