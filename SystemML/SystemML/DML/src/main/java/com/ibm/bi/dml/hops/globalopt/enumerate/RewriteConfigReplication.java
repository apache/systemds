/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.InterestingPropertyType;


/**
 * 
 */
public class RewriteConfigReplication extends RewriteConfig
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//valid instance configurations
	private static int[] _defValues = GlobalEnumerationOptimizer.REPLICATION_FACTORS;  

	public RewriteConfigReplication()
	{
		super( RewriteConfigType.REPLICATION_FACTOR, -1 );
	}
	
	@Override
	public int[] getDefinedValues()
	{
		return _defValues;
	}

	@Override
	public InterestingProperty getInterestingProperty()
	{
		//direct mapping from rewrite config to interesting property
		return new InterestingProperty(InterestingPropertyType.REPLICATION, getValue());
	}
	
/*

	@Override
	public void applyToHop(Hop hop) {
		hop.setColsInBlock(this.value);
		hop.setRowsInBlock(this.value);
	}

	@Override
	public boolean isValidForOperator(Hop operator) {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public String getValueString() {
		return String.valueOf(this.getValue());
	}

	@Override
	public RewriteConfig extractParamFromHop(Hop hop) {
		//TODO: rectangular blocksize
		Integer extractedBlockSize = (int)hop.getRowsInBlock();
		RewriteConfig extracted = this.createInstance(extractedBlockSize);
		return extracted;
	}
*/

}
