package dml.parser;

/**
 * <p>Defines a parameter of type double with its name and value.</p>
 * 
 * @author schnetter
 */
public class GenericFunctionDoubleParam extends GenericFunctionParam
{
	/** value of the double parameter */
	private double _paramValue;
	
	
	/**
	 * <p>Creates a new parameter of type double.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	public GenericFunctionDoubleParam(String paramName, double paramValue)
	{
		super(paramName, ParamType.DOUBLE);
	}
	
	/**
	 * <p>Returns the value of the parameter.</p>
	 * 
	 * @return parameter value
	 */
	public double getParamValue()
	{
		return _paramValue;
	}
	
	/**
	 * <p>Returns a string representation of the parameter.</p>
	 */
	public String toString()
	{
		return _paramName + "=" + Double.toString(_paramValue);
	}
}
