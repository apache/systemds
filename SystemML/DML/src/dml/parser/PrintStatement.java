package dml.parser;

import dml.utils.LanguageException;
 
public class PrintStatement extends Statement{
	protected Expression _expr;

	public PrintStatement(Expression expr){
		_expr = expr; 
	}
	 
	public Statement rewriteStatement(String prefix) throws LanguageException{
		Expression newExpr = _expr.rewriteExpression(prefix);
		return new PrintStatement(newExpr);
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	/*
	public String toString(){
		 StringBuffer sb = new StringBuffer();
		 boolean first = true;
		 sb.append(Statement.PRINTSTATEMENT + " ( " );
		 if (_msg != null){
			 sb.append(_msg);
			 first = false;
		 }
		 if (_id != null){
			 if (!first){
				 sb.append(",");
			 }
			 sb.append(_id.toString());
		 }
		 
		 sb.append(");");
		 return sb.toString(); 
	}
	*/
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result =  _expr.variablesRead();
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		 
	  	return new VariableSet();
	}

	@Override
	public boolean controlStatement() {
		 
		return false;
	}

	public Expression getExpression(){
		return _expr;
	}	
	 
}
