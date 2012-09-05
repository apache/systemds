package com.ibm.bi.dml.parser;

public class ConditionalPredicate {
	Expression _expr;
	
	
	public ConditionalPredicate(Expression expr){
		_expr = expr;
	}
	
	public Expression getPredicate(){
		return _expr;
	}
	
	public String toString(){
		return _expr.toString();
	}
	
	 
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariables(_expr.variablesRead());
	 	return result;
	}

	 
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariables(_expr.variablesUpdated());
	 	return result;
	}
	
	///////////////////////////////////////////////////////////////////////////
	// store position information for expressions
	///////////////////////////////////////////////////////////////////////////
	public int _beginLine, _beginColumn;
	public int _endLine, _endColumn;
	
	public void setBeginLine(int passed)    { _beginLine = passed;   }
	public void setBeginColumn(int passed) 	{ _beginColumn = passed; }
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	
	public void setAllPositions(int blp, int bcp, int elp, int ecp){
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	
	public String printErrorLocation(){
		return "ERROR: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	
}
