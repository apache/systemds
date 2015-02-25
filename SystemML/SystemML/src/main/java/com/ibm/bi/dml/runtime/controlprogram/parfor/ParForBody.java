/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
