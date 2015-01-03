/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.debug.DMLBreakpointManager;


public class AssignmentStatement extends Statement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private ArrayList<DataIdentifier> _targetList;
	private Expression _source;
	 
	// rewrites statement to support function inlining (creates deep copy)
	public Statement rewriteStatement(String prefix) throws LanguageException{
				
		// rewrite target (deep copy)
		DataIdentifier newTarget = (DataIdentifier)_targetList.get(0).rewriteExpression(prefix);
		
		// rewrite source (deep copy)
		Expression newSource = _source.rewriteExpression(prefix);
		
		// create rewritten assignment statement (deep copy)
		AssignmentStatement retVal = new AssignmentStatement(newTarget, newSource,this.getBeginLine(), 
											this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		return retVal;
	}
	
	
	public AssignmentStatement(DataIdentifier t, Expression s) {

		_targetList = new ArrayList<DataIdentifier>();
		_targetList.add(t);
		_source = s;
	}
	
	
	public AssignmentStatement(DataIdentifier t, Expression s, int beginLine, int beginCol, int endLine, int endCol) throws LanguageException{
		
		_targetList = new ArrayList<DataIdentifier>();
		_targetList.add(t);
		_source = s;
	
		setBeginLine(beginLine);
		setBeginColumn(beginCol);
		setEndLine(endLine);
		setEndColumn(endCol);
		
	}
	
	public DataIdentifier getTarget(){
		return _targetList.get(0);
	}
	
	public ArrayList<DataIdentifier> getTargetList()
	{
		return _targetList;
	}

	public Expression getSource(){
		return _source;
	}
	public void setSource(Expression s){
		_source = s;
	}
	
	@Override
	public boolean controlStatement() {
		// ensure that breakpoints end up in own statement block 
		if (DMLScript.ENABLE_DEBUG_MODE) {
			DMLBreakpointManager.insertBreakpoint(_source.getBeginLine());
			return true;
		}

		// for now, ensure that function call ends up in different statement block
		if (_source instanceof FunctionCallIdentifier)
			return true;
		
		return false;
	}
	
	public void initializeforwardLV(VariableSet activeIn){
		//do nothing
	}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		// add variables read by source expression
		result.addVariables(_source.variablesRead());
		
		// for LHS IndexedIdentifier, add variables for indexing expressions
		for (int i=0; i<_targetList.size(); i++){
			if (_targetList.get(i) instanceof IndexedIdentifier) {
				IndexedIdentifier target = (IndexedIdentifier) _targetList.get(i);
				result.addVariables(target.variablesRead());
			}
		}		
		return result;
	}
	
	public  VariableSet variablesUpdated() {
		VariableSet result =  new VariableSet();
		
		// add target to updated list
		for (DataIdentifier target : _targetList)
			result.addVariable(target.getName(), target);
		return result;
	}
	
	public String toString(){
		String retVal  = "";
		for (int i=0; i< _targetList.size(); i++){
			retVal += _targetList.get(i).toString();
		}
		retVal += " = " + _source.toString() + ";";
		return retVal;
	}
}
