/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

public class GridEnumerationExp extends GridEnumeration
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final double DEFAULT_FACTOR = 2;

	private double _factor = -1;
	
	public GridEnumerationExp( ArrayList<ProgramBlock> prog, long min, long max ) 
		throws DMLRuntimeException
	{
		super(prog, min, max);
		
		_factor = DEFAULT_FACTOR;
	}
	
	/**
	 * 
	 * @param steps
	 */
	public void setFactor( double factor )
	{
		_factor = factor;
	}
	
	@Override
	public ArrayList<Long> enumerateGridPoints() 
	{
		ArrayList<Long> ret = new ArrayList<Long>();
		long v = _min;
		while( v < _max ) {
			ret.add( v );
			v *= _factor;
		}
		ret.add(_max);
		
		return ret;
	}
}
