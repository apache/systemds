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

public class GridEnumerationEqui extends GridEnumeration
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final int DEFAULT_NSTEPS = 20;

	private int _nsteps = -1;
	
	public GridEnumerationEqui( ArrayList<ProgramBlock> prog, double min, double max ) 
		throws DMLRuntimeException
	{
		super(prog, min, max);
		
		_nsteps = DEFAULT_NSTEPS;
	}
	
	/**
	 * 
	 * @param steps
	 */
	public void setNumSteps( int steps )
	{
		_nsteps = steps;
	}
	
	@Override
	public ArrayList<Double> enumerateGridPoints() 
	{
		ArrayList<Double> ret = new ArrayList<Double>();
		double gap = (_max - _min) / (_nsteps-1);
		double v = _min;
		for (int i = 0; i < _nsteps; i++) {
			ret.add(v);
			v += gap;
		}
		return ret;
	}
}
