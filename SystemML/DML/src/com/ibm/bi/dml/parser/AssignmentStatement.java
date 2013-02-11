package com.ibm.bi.dml.parser;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.utils.LanguageException;


public class AssignmentStatement extends Statement{
	
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
	
	public AssignmentStatement(DataIdentifier t, Expression s){

		_targetList = new ArrayList<DataIdentifier>();
		_targetList.add(t);
		_source = s;
	}
	
	public AssignmentStatement(DataIdentifier t, Expression s, int beginLine, int beginCol, int endLine, int endCol){
		
		_targetList = new ArrayList<DataIdentifier>();
		_targetList.add(t);
		_source = s;
	
		_beginLine   = beginLine;
		_beginColumn = beginCol;
		_endLine     = endLine;
		_endColumn   = endCol;
		
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
		// for now, ensure that function call ends up in different statement block
		if (_source instanceof FunctionCallIdentifier)
			return true;
		
		// ensure that specific ops end up in different statement block
		if( containsIndividualStatementBlockOperations() )
			return true;
		
		return false;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean containsIndividualStatementBlockOperations()
	{
		boolean ret = false;
		
		if( OptimizerUtils.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS )
		{
			//TODO
			//if( _source.toString().contains(Expression.DataOp.RAND.toString()) )
			//	ret = true;	
			
			//TODO enable this for groupedAggregate after resolved reblock issue
			//if( _source.toString().contains(Expression.ParameterizedBuiltinFunctionOp.GROUPEDAGG.toString()) )
			//	ret = true;	
			
			if( _source.toString().contains(Expression.BuiltinFunctionOp.CTABLE.toString()) ) 
				ret = true;
			

		}
		//System.out.println(_source +": "+ret);
		
		return ret;
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
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
