package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.utils.LanguageException;

 
public class OutputStatement extends IOStatement{
	
	public OutputStatement(DataIdentifier t, String fname){
		super(t,null);
	}
	public Statement rewriteStatement(String prefix) throws LanguageException{
		
		OutputStatement newStatement = new OutputStatement(this);
		String newIdName = prefix + this._id.getName();
		newStatement.getId().setName(newIdName);
		return newStatement;
	}
	
	public OutputStatement(OutputStatement ostmt){
		_id           = ostmt._id;
		_filename     = ostmt._filename;
		_varParams   = ostmt._varParams;
		_stringParams = ostmt._stringParams;
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	public String toString(){
		 StringBuffer sb = new StringBuffer();
		 sb.append(Statement.OUTPUTSTATEMENT + " ( " );
		 sb.append( _id.toString() + "," + "\"" + _filename + "\"");
		 for (String key : _stringParams.keySet()){
			 sb.append("," + key + "=" + "\"" + _stringParams.get(key) + "\"");
		 }
		 for (String key : _varParams.keySet()){
			 sb.append("," + key + "=" + _varParams.get(key));
		 }
		 sb.append(");");
		 return sb.toString(); 
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		result.addVariable(_id.getName(), _id);
		
		for (String key : _varParams.keySet()){
			DataIdentifier id = new DataIdentifier(_varParams.get(key));
			result.addVariable(_varParams.get(key), id) ;
		}
		
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return null;
	}
	
	 
}
