/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;



/**
 * 
 */
public class RewriteConfigVectorization extends RewriteConfig
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//valid instance configurations
	private static int[] _defValues = new int[]{ 0, 1 }; //no | yes  
	
	public RewriteConfigVectorization()
	{
		super( RewriteConfigType.VECTORIZATION, -1 );
	}
	
	@Override
	public int[] getDefinedValues()
	{
		return _defValues;
	}

	@Override
	public InterestingProperty getInterestingProperty()
	{
		//no interesting property directly influenced here
		//however, interesting properties of input determine if valid and if cost beneficial
		return null;
	}
}
