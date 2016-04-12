/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.api.jmlc;

import java.util.HashMap;
import java.util.Set;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;

/**
 * A ResultVariables object holds the data returned by a call
 * to {@link PreparedScript}'s {@link PreparedScript#executeScript executeScript}
 * method, which executes a DML/PyDML script.
 *
 */
public class ResultVariables 
{
	
	private HashMap<String, Data> _out = null;
	
	public ResultVariables()
	{
		_out = new HashMap<String, Data>();
	}
	
	/**
	 * Obtain the output variable names held by this ResultVariables object.
	 * 
	 * @return the set of output variable names
	 */
	public Set<String> getVariableNames()
	{
		return _out.keySet();
	}
	
	/**
	 * Obtain the number of output data held by this ResultVariables object.
	 * 
	 * @return the number of output variables with data
	 */
	public int size()
	{
		return _out.size();
	}
	
	/**
	 * Obtain the matrix represented by the given output variable.
	 * 
	 * @param varname output variable name
	 * @return matrix as a two-dimensional double array
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
	 * Obtain the frame represented by the given output variable.
	 * 
	 * @param varname output variable name
	 * @return frame as a two-dimensional string array
	 * @throws DMLException
	 */
	public String[][] getFrame(String varname) 
		throws DMLException
	{
		if( !_out.containsKey(varname) )
			throw new DMLException("Non-existing output variable: "+varname);
		
		String[][] ret = null;
		Data dat = _out.get(varname);
		
		//basic checks for data type	
		if( !(dat instanceof FrameObject) )
			throw new DMLException("Expected frame result '"+varname+"' not a frame.");
		
		//convert output matrix to double array	
		FrameObject fo = (FrameObject)dat;
		FrameBlock frame = fo.acquireRead();
		ret = DataConverter.convertToStringFrame(frame);
		fo.release();
		
		return ret;
	}
	
	/**
	 * Add the output variable name and generated output data to the ResultVariable
	 * object. Called during the execution of {@link PreparedScript}'s
	 * {@link PreparedScript#executeScript executeScript} method.
	 * 
	 * @param ovar output variable name
	 * @param data generated output data
	 */
	protected void addResult(String ovar, Data data) 
	{
		_out.put(ovar, data);
	}
}
