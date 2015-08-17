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

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;

/**
 * Wrapper for exchanging parfor body data structures.
 * 
 */
public class ParForBody 
{

	
	private ArrayList<String>       _resultVarNames;
	private ArrayList<ProgramBlock> _childBlocks;
	private ExecutionContext 		_ec;
	
	public ParForBody()
	{
		
	}

	public ParForBody( ArrayList<ProgramBlock> childBlocks, 
			ArrayList<String> resultVarNames, ExecutionContext ec ) 
	{
		_resultVarNames = resultVarNames;
		_childBlocks    = childBlocks;
		_ec             = ec;
	}

	public LocalVariableMap getVariables() 
	{
		return _ec.getVariables();
	}

	public ArrayList<String> getResultVarNames() 
	{
		return _resultVarNames;
	}

	public void setResultVarNames(ArrayList<String> resultVarNames) 
	{
		_resultVarNames = resultVarNames;
	}

	public ArrayList<ProgramBlock> getChildBlocks() 
	{
		return _childBlocks;
	}

	public void setChildBlocks(ArrayList<ProgramBlock> childBlocks) 
	{
		_childBlocks = childBlocks;
	}

	public ExecutionContext getEc() 
	{
		return _ec;
	}

	public void setEc(ExecutionContext ec) 
	{
		_ec = ec;
	}
}
