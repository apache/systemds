package dml.parser;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import dml.utils.LanguageException;

public class ExternalFunctionStatement extends FunctionStatement{
	
	private HashMap<String,String> _otherParams;
	
	public ExternalFunctionStatement(){
		super();
	}
	
	public void setOtherParams(HashMap<String,String> params){
		_otherParams = params;
	}
	public HashMap<String,String> getOtherParams(){
		return _otherParams;
	}
	
	@Override
	public boolean controlStatement() {
		return true;
	}
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		sb.append(_name + " = ");
		
		sb.append("externalfunction ( ");
		for (int i=0; i<_inputParams.size(); i++){
			DataIdentifier curr = _inputParams.get(i);
			sb.append(curr.getName());
			if (curr.getDefaultValue() != null) sb.append(" = " + curr.getDefaultValue());
			if (i < _inputParams.size()-1) sb.append(", ");
		}
		sb.append(") return (");
		
		for (int i=0; i<_outputParams.size(); i++){
			sb.append(_outputParams.get(i).getName());
			if (i < _outputParams.size()-1) sb.append(", ");
		}
		sb.append(")\n implemented in (");
		
		int numOtherParams = _otherParams.keySet().size();
		int j = 0;
		for (String key : _otherParams.keySet()){
			sb.append(key + " = " + _otherParams.get(key));
			if (j < _otherParams.keySet().size()-1) sb.append(", ");
			j++;
		}
		
		sb.append(") \n");
		return sb.toString();
	}

	public void initializeforwardLV(VariableSet activeIn) throws LanguageException{
		throw new LanguageException("should never call initializeforwardLV for FunctionStatement");
	}
	
	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		throw new LanguageException("should never call initializeforwardLV for FunctionStatement");
		
	}
	
	@Override
	public VariableSet variablesRead() {
		System.out.println("[W] should not call variablesRead from FunctionStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		System.out.println("[W] should not call variablesRead from FunctionStatement ");
		return new VariableSet();
	}
}
