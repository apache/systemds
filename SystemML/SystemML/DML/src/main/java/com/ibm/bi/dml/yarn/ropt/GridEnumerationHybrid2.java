/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

/**
 * Composite overlay of hybrid and exp grid.
 * 
 */
public class GridEnumerationHybrid2 extends GridEnumeration
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public GridEnumerationHybrid2( ArrayList<ProgramBlock> prog, long min, long max ) 
		throws DMLRuntimeException
	{
		super(prog, min, max);
	}
	
	@Override
	public ArrayList<Long> enumerateGridPoints() 
		throws DMLRuntimeException, HopsException
	{
		GridEnumeration ge1 = new GridEnumerationHybrid(_prog, _min, _max);
		GridEnumeration ge2 = new GridEnumerationExp(_prog, _min, _max);
		
		//ensure distinct points
		HashSet<Long> hs = new HashSet<Long>();
		hs.addAll( ge1.enumerateGridPoints() );
		hs.addAll( ge2.enumerateGridPoints() );
		
		//create sorted output list
		ArrayList<Long> ret = new ArrayList<Long>();
		for( Long val : hs )
			ret.add(val);
		Collections.sort(ret); //asc
		
		return ret;
	}
}
