package dml.parser;

import java.util.HashMap;

import dml.utils.LanguageException;
 
public class InputStatement extends IOStatement{
		
	public Statement rewriteStatement(String prefix) throws LanguageException{
		
		InputStatement newStatement = new InputStatement(this);
		String newIdName = prefix + this._id.getName();
		newStatement.getId().setName(newIdName);
		
		for (String key : _varParams.keySet()){
			String newName = prefix + _varParams.get(key);
			_varParams.put(key, newName);
		}
		
		return newStatement;
	}

	public InputStatement(){
		super();
	}
	
	public InputStatement(InputStatement istmt){
		_id           = istmt._id;
		_filename     = istmt._filename;
		_stringParams = istmt._stringParams;
		_varParams   = istmt._varParams;
	
	}
	
	public InputStatement(DataIdentifier t, String fname){
		super(t,fname);
	}

	public void initializeforwardLV(VariableSet activeIn){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}

	
	public String toString(){
		 StringBuffer sb = new StringBuffer();
		 sb.append(_id.toString() + " = " + Statement.INPUTSTATEMENT + " ( " );
		 sb.append("\""+_filename+"\"");
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
		for (String key : _varParams.keySet()){
			result.addVariable(_varParams.get(key), new DataIdentifier(_varParams.get(key))) ;
		}
		
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		result.addVariable(_id.getName(),_id);
	 	return result;
	}
}
