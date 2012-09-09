package com.ibm.bi.dml.parser;

import java.util.HashMap;

import com.ibm.bi.dml.utils.LanguageException;


public class ExternalFunctionStatement extends FunctionStatement{
	
	//valid attribute names
	public static final String CLASS_NAME    = "classname";
	public static final String EXEC_TYPE     = "exectype";
	public static final String EXEC_LOCATION = "execlocation";
	public static final String CONFIG_FILE   = "configfile";
	
	//valid attribute values for execlocation and 
	public static final String FILE_BASED    = "file";
	public static final String IN_MEMORY     = "mem";
	public static final String MASTER        = "master";
	public static final String WORKER        = "worker";
	
	//default values for optional attributes
	public static final String DEFAULT_EXEC_TYPE = FILE_BASED;
	public static final String DEFAULT_EXEC_LOCATION = MASTER;
	
	//all parameters
	private HashMap<String,String> _otherParams;
	
	
	public ExternalFunctionStatement()
	{
		super();
	}
	
	public void setOtherParams(HashMap<String,String> params)
	{
		_otherParams = params;
	}
	
	public HashMap<String,String> getOtherParams()
	{
		return _otherParams;
	}
	
	/**
	 * Validates all attributes and attribute values.
	 * 
	 * @throws LanguageException
	 */
	public void validateParameters()
		throws LanguageException
	{
		//warnings for all not defined attributes
		for( String varName : _otherParams.keySet() )
			if( !(   varName.equals(CLASS_NAME) || varName.equals(EXEC_TYPE) || varName.equals(EXEC_LOCATION) 
				  || varName.equals(CONFIG_FILE) ) )                                                  
				System.out.println("WARNING: line " + this.getBeginLine() + ", column " + this.getBeginColumn() + " -- External function specifies undefined attribute type '"+varName+"'.");
		
		//class name (required)
		if( !_otherParams.containsKey(CLASS_NAME) )
			throw new LanguageException(this.printErrorLocation() + "External function does not specify the required attribute '"+CLASS_NAME+"'.");
		else if ( _otherParams.get(CLASS_NAME)==null )
			throw new LanguageException(this.printErrorLocation() + "External function specifies empty '"+CLASS_NAME+"'.");
		
		//exec type (optional, default: file)
		if( _otherParams.containsKey( EXEC_TYPE ) )
		{
			//check specified values
			String execType = _otherParams.get(EXEC_TYPE);
			if( !(execType.equals(FILE_BASED) || execType.equals(IN_MEMORY)) )
				throw new LanguageException(this.printErrorLocation() + "External function specifies invalid value for (optional) attribute '"+EXEC_TYPE+"' (valid values: "+FILE_BASED+","+IN_MEMORY+").");
		}
		else
		{
			//put default values
			_otherParams.put(EXEC_TYPE, DEFAULT_EXEC_TYPE);
		}
		
		//exec location (optional, default: master)
		if( _otherParams.containsKey( EXEC_LOCATION ) )
		{
			//check specified values
			String execLocation = _otherParams.get(EXEC_LOCATION);
			if( !(execLocation.equals(MASTER) || execLocation.equals(WORKER)) )
				throw new LanguageException(this.printErrorLocation() + "External function specifies invalid value for (optional) attribute '"+EXEC_LOCATION+"' (valid values: "+MASTER+","+WORKER+").");
		}
		else
		{
			//put default values
			_otherParams.put(EXEC_LOCATION, DEFAULT_EXEC_LOCATION);
		}
	}
	
	@Override
	public boolean controlStatement() 
	{
		return true;
	}
	
	@Override
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
		
		int j = 0;
		for (String key : _otherParams.keySet()){
			sb.append(key + " = " + _otherParams.get(key));
			if (j < _otherParams.keySet().size()-1) sb.append(", ");
			j++;
		}
		
		sb.append(") \n");
		return sb.toString();
	}

	@Override
	public void initializeforwardLV(VariableSet activeIn) throws LanguageException{
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for ExternalFunctionStatement");
	}
	
	@Override
	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for ExternalFunctionStatement");
		
	}
	
	@Override
	public VariableSet variablesRead() {
		System.out.println(this.printWarningLocation() + "should not call variablesRead from ExternalFunctionStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		System.out.println(this.printWarningLocation() + "should not call variablesRead from ExternalFunctionStatement ");
		return new VariableSet();
	}
}
