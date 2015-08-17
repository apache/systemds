/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

/**
 * Class to store metadata associated with a file (e.g., a matrix) on disk.
 * This class must be extended to associate specific information with the file. 
 *
 */

public abstract class MetaData
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
			
	@Override
	public abstract boolean equals (Object anObject);
	
	@Override
	public int hashCode()
	{
		//use identity hash code
		return super.hashCode();
	}
	
	@Override
	public abstract String toString();
	
	@Override 
	public abstract Object clone();
}
