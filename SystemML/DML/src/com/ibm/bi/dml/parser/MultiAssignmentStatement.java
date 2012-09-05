package com.ibm.bi.dml.parser;

import java.util.ArrayList;

import com.ibm.bi.dml.utils.LanguageException;


public class MultiAssignmentStatement extends Statement{
	
	private ArrayList<DataIdentifier> _targetList;
	private Expression _source;
		
	// rewrites statement to support function inlining (creates deep copy) 
	public Statement rewriteStatement(String prefix) throws LanguageException{
				
		ArrayList<DataIdentifier> newTargetList = new ArrayList<DataIdentifier>();
		
		// rewrite targetList (deep copy)
		for (DataIdentifier target : _targetList){
			DataIdentifier newTarget = (DataIdentifier) target.rewriteExpression(prefix);
			newTargetList.add(newTarget);
		}
		
		// rewrite source (deep copy)
		Expression newSource = _source.rewriteExpression(prefix);
		
		// create rewritten assignment statement (deep copy)
		MultiAssignmentStatement retVal = new MultiAssignmentStatement(newTargetList, newSource);
		retVal._beginLine   = this.getBeginLine();
		retVal._beginColumn = this.getBeginColumn();
		retVal._endLine 	= this.getEndLine();
		retVal._endColumn	= this.getEndColumn();
		
		return retVal;
	}
	
	public MultiAssignmentStatement(ArrayList<DataIdentifier> tList, Expression s){
		_targetList = tList;
		_source = s;
	}
	
	// NOTE: f is not used -- however, error is thrown "methods have same erasure" if not included in signature
	public MultiAssignmentStatement(ArrayList<ArrayList<Expression>> exprListList, Expression s, int f){
		
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
		VariableSet result = new VariableSet();
		
		// add variables read by source expression
		result.addVariables(_source.variablesRead());
		
		// for any IndexedIdentifier on LHS, add variables for indexing expressions
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
