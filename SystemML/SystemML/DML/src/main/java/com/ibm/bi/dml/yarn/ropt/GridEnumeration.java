/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

public abstract class GridEnumeration 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected ArrayList<ProgramBlock> _prog = null;
	protected long  _min = -1;
	protected long  _max = -1;

	public GridEnumeration( ArrayList<ProgramBlock> prog, long min, long max ) 
		throws DMLRuntimeException
	{
		if( min > max )
			throw new DMLRuntimeException("Invalid parameters: min=" + min + ", max=" + max);
		
		_prog = prog;
		_min = min;
		_max = max;
	}
	
	/**
	 * 
	 * @return
	 * @throws HopsException 
	 * @throws DMLException
	 */
	public abstract ArrayList<Long> enumerateGridPoints() 
		throws DMLRuntimeException, HopsException; 
}
