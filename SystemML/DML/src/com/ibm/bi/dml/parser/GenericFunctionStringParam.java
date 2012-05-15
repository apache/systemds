package com.ibm.bi.dml.parser;

/**
 * <p>Defines a parameter of type string with its name and value.</p>
 * 
 * @author schnetter
 */
public class GenericFunctionStringParam extends GenericFunctionParam
{
	/** value of the string parameter */
	private String _paramValue;
	
	
	/**
	 * <p>Creates a new parameter of type string.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	public GenericFunctionStringParam(String paramName, String paramValue)
	{
		super(paramName, ParamType.STRING);
		_paramValue = paramValue;
	}
	
	/**
	 * <p>Returns the value of the parameter.</p>
	 * 
	 * @return parameter value
	 */
	public String getParamValue()
	{
		return _paramValue;
	}
	
	/**
	 * <p>Returns a string representation of the parameter.</p>
	 */
	public String toString()
	{
		return _paramName + "=\"" + _paramValue + "\"";
	}
}
