/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
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
