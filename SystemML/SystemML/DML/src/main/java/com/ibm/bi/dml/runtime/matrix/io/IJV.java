/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.io;

/**
 * Helper class for external key/value exchange.
 * 
 */
public class IJV
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public int i=-1;
	public int j=-1;
	public double v=0;
	
	public IJV()
	{
		
	}
	
	public IJV(int i, int j, double v)
	{
		set(i, j, v);
	}
	
	public void set(int i, int j, double v)
	{
		this.i = i;
		this.j = j;
		this.v = v;
	}
	
	public String toString()
	{
		return "("+i+", "+j+"): "+v;
	}
}
