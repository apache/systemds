/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import java.io.IOException;

/**
 * Any data output that is intended to support fast serialization / write
 * of entire blocks should implement this interface. On write of a matrix block
 * we check if the output stream is an implementation of this interface, if
 * yes we directly pass the entire block instead of value-by-value.
 * 
 * Known implementation classes:
 *    - CacheDataOutput (cache serialization into in-memory write buffer)
 *    - FastBufferedDataOutputStream (cache eviction to local file system)
 * 
 */
public interface MatrixBlockDataOutput 
{
	/**
	 * Writes the double array of a dense block to the data output. 
	 * 
	 * @param len
	 * @param varr
	 * @throws IOException
	 */
	public void writeDoubleArray(int len, double[] varr) 
		throws IOException;
	
	/**
	 * Writes the sparse rows array of a sparse block to the data output.
	 * 
	 * @param rlen
	 * @param rows
	 * @throws IOException
	 */
	public void writeSparseRows(int rlen, SparseRow[] rows) 
		throws IOException;
}
