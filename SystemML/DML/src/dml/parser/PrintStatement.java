package dml.parser;

import java.util.HashMap;

import dml.utils.LanguageException;
 
public class PrintStatement extends Statement{
	protected DataIdentifier _id;
	protected String _msg;

	public PrintStatement(String msg,DataIdentifier id){
		_id = id;
		msg = msg.replaceAll(":", " ");
		_msg = msg;
	}
	 
	public Statement rewriteStatement(String prefix) throws LanguageException{
		DataIdentifier newId = new DataIdentifier(_id);
		String newIdName = prefix + _id.getName();
		newId.setName(newIdName);
		return new PrintStatement(_msg, newId);
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
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
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		if (_id != null){
			result.addVariable(_id.getName(),_id);
		}
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
	
	public DataIdentifier getIdentifier(){
		return _id;
	}
	
	public String getMessage(){
		return _msg;
	}
	
	 
}
