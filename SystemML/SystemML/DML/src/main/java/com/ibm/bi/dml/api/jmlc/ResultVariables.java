/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.api.jmlc;

import java.util.HashMap;
import java.util.Set;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.DataConverter;

/**
 * JMLC (Java Machine Learning Connector) API:
 * 
 * NOTE: Currently fused API and implementation in order to reduce complexity. 
 */
public class ResultVariables 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private HashMap<String, Data> _out = null;
	
	public ResultVariables()
	{
		_out = new HashMap<String, Data>();
	}
	
	public Set<String> getVariableNames()
	{
		return _out.keySet();
	}
	
	public int size()
	{
		return _out.size();
	}
	
	/**
	 * 
	 * @param var
	 * @return
	 * @throws DMLException
	 */
	public double[][] getMatrix(String varname) 
		throws DMLException
	{
		if( !_out.containsKey(varname) )
			throw new DMLException("Non-existing output variable: "+varname);
		
		double[][] ret = null;
		Data dat = _out.get(varname);
		
		//basic checks for data type	
		if( !(dat instanceof MatrixObject) )
			throw new DMLException("Expected matrix result '"+varname+"' not a matrix.");
		
		//convert output matrix to double array	
		MatrixObject mo = (MatrixObject)dat;
		MatrixBlock mb = mo.acquireRead();
		ret = DataConverter.convertToDoubleMatrix(mb);
		mo.release();
	
		return ret;
	}
	
	/**
	 * 
	 * 
	 * @param ovar
	 * @param data
	 */
	protected void addResult(String ovar, Data data) 
	{
		_out.put(ovar, data);
	}
}
