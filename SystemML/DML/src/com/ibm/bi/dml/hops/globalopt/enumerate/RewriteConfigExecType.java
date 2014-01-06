/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.DataLocationType;
import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.InterestingPropertyType;
import com.ibm.bi.dml.lops.LopProperties.ExecType;


/**
 * 
 */
public class RewriteConfigExecType extends RewriteConfig
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//valid instance configurations
	private static int[] _defValues = new int[]{ ExecType.CP.ordinal(), 
		                                         ExecType.MR.ordinal() };  
	
	public RewriteConfigExecType()
	{
		super( RewriteConfigType.EXEC_TYPE, -1 );
	}
	
	public RewriteConfigExecType(int value)
	{
		super( RewriteConfigType.EXEC_TYPE, value );
	}
	
	@Override
	public int[] getDefinedValues()
	{
		return _defValues;
	}

	@Override
	public InterestingProperty getInterestingProperty()
	{
		// mapping from rewrite config exec type to interesting property data location
		ExecType et = ExecType.values()[getValue()];
		DataLocationType dlt = null;
		switch( et )
		{
			case CP: dlt = DataLocationType.MEM; break;
			case MR: dlt = DataLocationType.HDFS; break;
			default: throw new RuntimeException("Unknown exec type: "+et);
		}
		
		return new InterestingProperty(InterestingPropertyType.DATA_LOCATION, dlt.ordinal());
	}
}
