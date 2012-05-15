package com.ibm.bi.dml.parser;

/**
 * <p>Defines a parameter of type variable with its name and value.</p>
 * 
 * @author schnetter
 */
public class GenericFunctionVarParam extends GenericFunctionParam
{
	/** value of the variable parameter */
	private DataIdentifier _paramValue;
	
	
	/**
	 * <p>Creates a new parameter of type variable.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	public GenericFunctionVarParam(String paramName, DataIdentifier paramValue)
	{
		super(paramName, ParamType.VAR);
		_paramValue = paramValue;
	}
	
	/**
	 * <p>Returns the value of the parameter.</p>
	 * 
	 * @return parameter value
	 */
	public DataIdentifier getParamValue()
	{
		return _paramValue;
	}
	
	/**
	 * <p>Returns a string representation of the parameter.</p>
	 */
	public String toString()
	{
		return _paramName + "=" + _paramValue.toString();
	}
}
