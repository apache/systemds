package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;

/**
 * Wrapper for exchanging parfor body datastructures.
 * 
 * @author mboehm
 */
public class ParForBody 
{
	private HashMap<String, Data> 		_variables;
	private ArrayList<String>           _resultVarNames;
	private ArrayList<ProgramBlock> 	_childBlocks;
	private ExecutionContext 			_ec;
	
	public ParForBody()
	{
		
	}

	public ParForBody( ArrayList<ProgramBlock> childBlocks, 
			           HashMap<String, Data> variables, ArrayList<String> resultVarNames, ExecutionContext ec ) 
	{
		_variables      = variables;
		_resultVarNames = resultVarNames;
		_childBlocks    = childBlocks;
		_ec             = ec;
	}

	public HashMap<String, Data> getVariables() 
	{
		return _variables;
	}

	public void setVariables(HashMap<String, Data> variables) 
	{
		_variables = variables;
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
