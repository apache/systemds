package com.ibm.bi.dml.runtime.matrix;

/**
 * Class to store metadata associated with a file (e.g., a matrix) on disk.
 * This class must be extended to associate specific information with the file. 
 *
 */

public abstract class MetaData
{
	
	@Override
	public abstract boolean equals (Object anObject);
	
	@Override
	public abstract String toString();
	
}
