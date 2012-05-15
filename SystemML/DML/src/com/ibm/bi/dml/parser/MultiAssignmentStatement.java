package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.utils.LanguageException;


public class MultiAssignmentStatement extends Statement{
	
	private ArrayList<DataIdentifier> _targetList;
	private Expression _source;
	
	// create a copy that has rewritten values for 
	public Statement rewriteStatement(String prefix) throws LanguageException{
				
		ArrayList<DataIdentifier> newTargetList = new ArrayList<DataIdentifier>();
		
		// rewrite targetList 
		for (DataIdentifier target : _targetList){
			String newTargetName = prefix + target.getName();	
			DataIdentifier newTarget = new DataIdentifier(target);
			newTarget.setName(newTargetName);	
			newTargetList.add(newTarget);
		}
		
		// rewrite source
		Expression newSource = _source.rewriteExpression(prefix);
		
		// create rewritten assignment statement 
		MultiAssignmentStatement retVal = new MultiAssignmentStatement(newTargetList, newSource);
		
		return retVal;
	}
	
	public MultiAssignmentStatement(ArrayList<DataIdentifier> tList, Expression s){
		_targetList = tList;
		_source = s;
	}
	
	public MultiAssignmentStatement(ArrayList<ArrayList<Expression>> exprListList, Expression s, int f){
		// f is not used -- however, error is thrown "methods have same erasure" is not included
		_source = s;
		
		_targetList = new ArrayList<DataIdentifier>();
		for (ArrayList<Expression> exprList : exprListList){
			Expression expr = exprList.get(0);
			_targetList.add(new DataIdentifier(expr.toString()));
		}
	}
	
	public ArrayList<DataIdentifier> getTargetList(){
		return _targetList;
	}
	
	public Expression getSource(){
		return _source;
	}
	
	@Override
	// conservative assignment to separate statement block; will merge later if possible
	public boolean controlStatement() {
		return true;
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	public VariableSet variablesRead() {
		VariableSet result = _source.variablesRead();
		return result;
	}
	
	public  VariableSet variablesUpdated() {
		VariableSet result =  new VariableSet();
		for (DataIdentifier target : _targetList){
			result.addVariable(target.getName(), target);
		}
		return result;
	}
	
	public String toString(){
		String retVal  = "[";
		for (int i=0; i< _targetList.size(); i++){
			retVal += _targetList.get(i).toString();
			if (i < _targetList.size() - 1)
				retVal += ",";
		}
		retVal += "] = " + _source.toString() + ";";
		return retVal;
	}

	public void setSource(FunctionCallIdentifier s) {
		_source = s;
		
	}
}
