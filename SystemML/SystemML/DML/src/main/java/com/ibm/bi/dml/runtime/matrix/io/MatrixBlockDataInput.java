/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

import java.io.IOException;

/**
 * Any data input that is intended to support fast deserialization / read
 * of entire blocks should implement this interface. On read of a matrix block
 * we check if the input stream is an implementation of this interface, if
 * yes we let the implementation directly pass the entire block instead of value-by-value.
 * 
 * Known implementation classes:
 *    TODO
 *    
 */
public interface MatrixBlockDataInput 
{
	/**
	 * Reads the double array from the data input into the given dense block
	 * and returns the number of non-zeros. 
	 * 
	 * @param len
	 * @param varr
	 * @return
	 * @throws IOException
	 */
	public int readDoubleArray(int len, double[] varr) 
		throws IOException;
	
	/**
	 * Reads the sparse rows array from the data input into a sparse block
	 * and returns the number of non-zeros.
	 * 
	 * @param rlen
	 * @param rows
	 * @throws IOException
	 */
	public int readSparseRows(int rlen, SparseRow[] rows) 
		throws IOException;
}
