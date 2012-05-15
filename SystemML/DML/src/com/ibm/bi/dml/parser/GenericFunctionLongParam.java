package com.ibm.bi.dml.parser;

/**
 * <p>Defines a parameter of type long with its name and value.</p>
 * 
 * @author schnetter
 */
public class GenericFunctionLongParam extends GenericFunctionParam
{
	/** value of the long parameter */
	private long _paramValue;
	
	
	/**
	 * <p>Creates a new parameter of type long.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	public GenericFunctionLongParam(String paramName, long paramValue)
	{
		super(paramName, ParamType.INT);
		_paramValue = paramValue;
	}
	
	/**
	 * <p>Returns the value of the parameter.</p>
	 * 
	 * @return parameter value
	 */
	public long getParamValue()
	{
		return _paramValue;
	}
	
	/**
	 * <p>Returns a string representation of the parameter.</p>
	 */
	public String toString()
	{
		return _paramName + "=" + Long.toString(_paramValue);
	}
}
