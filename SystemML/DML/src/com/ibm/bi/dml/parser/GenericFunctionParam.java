package com.ibm.bi.dml.parser;

/**
 * <p>Defines a parameter with its type, name and value.</p>
 * 
 * @author schnetter
 */
public abstract class GenericFunctionParam
{
	/** available parameter types */
	public static enum ParamType { INT, DOUBLE, STRING, VAR };
	
	/** name of the parameter */
	protected String	_paramName;
	/** type of the parameter */
	protected ParamType	_paramType;
	
	
	/**
	 * <p>Creates a new parameter with the parameter name and type.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramType parameter type
	 */
	public GenericFunctionParam(String paramName, ParamType paramType)
	{
		_paramName	= paramName;
		_paramType	= paramType;
	}
	
	/**
	 * <p>Returns the name of the parameter.</p>
	 * 
	 * @return parameter name
	 */
	public String getParamName()
	{
		return _paramName;
	}
	
	/**
	 * <p>Returns the type of the parameter.</p>
	 * 
	 * @return parameter type
	 */
	public ParamType getParamType()
	{
		return _paramType;
	}
}
