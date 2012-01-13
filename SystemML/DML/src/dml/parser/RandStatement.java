package dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import dml.parser.Expression.FormatType;
import dml.parser.Expression.ValueType;
import dml.utils.LanguageException;

/**
 * <p>Defines a Rand-Statement.</p>
 * 
 * @author schnetter
 */
public class RandStatement extends Statement
{
	/**
	 * <p>Defines the list of available parameters and their DML component.</p>
	 * 
	 * @author schnetter
	 */
	private enum Param
	{
		/** number of rows */
		ROWS("rows"),
		/** number of columns */
		COLS("cols"),
		/** minimum of the random values */
		MIN("min"),
		/** maximum of the random values */
		MAX("max"),
		/** sparsity of the random object */
		SPARSITY("sparsity"),
		/** probability density function */
		PDF("pdf");
		
		/** holds the DML parameter name */
		private String paramName;
		
		
		/**
		 * <p>Adds a new parameter to the list of available ones.</p>
		 * 
		 * @param paramName DML parameter name
		 */
		private Param(String paramName)
		{
			this.paramName = paramName;
		}
		
		/**
		 * <p>Returns the DML parameter name.</p>
		 * 
		 * @return DML parameter name
		 */
		public String getParamName()
		{
			return paramName;
		}
	};
	
	/** target identifier which will hold the random object */
	private DataIdentifier _id;
	/** number of rows */
	private long rows = 1;
	/** number of columns */
	private long cols = 1;
	/** minimum of the random values */
	private double minValue = 0.0;
	/** maximum of the random values */
	private double maxValue = 1.0;
	/** sparsity of the random object */
	private double sparsity = 1.0;
	/** probability density function used to produce the sparsity */
	private String probabilityDensityFunction = "uniform";
	/** list of required variables */
	private HashMap<String, String> _requiredVariables;
	
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		
		RandStatement newStatement = new RandStatement(this);
		String newIdName = prefix + _id.getName();
		newStatement._id.setName(newIdName);
		return newStatement;
	}
	
	
	
	/**
	 * <p>Creates a new rand statement.</p>
	 */
	public RandStatement()
	{
		_id = null;
		_requiredVariables = new HashMap<String, String>();
	}
	
	public RandStatement(RandStatement rand)
	{
		DataIdentifier newId = new DataIdentifier(rand._id);
		HashMap<String, String> newRequiredVariables = new HashMap<String, String>();
		for (String key : rand._requiredVariables.keySet()){
			newRequiredVariables.put(key, rand._requiredVariables.get(key));
		}
		
		
		this._id = newId;
		this._requiredVariables = newRequiredVariables;
		
		
	}
	
	/**
	 * <p>Creates a new rand statement.</p>
	 * 
	 * @param id target identifier
	 */
	public RandStatement(DataIdentifier id)
	{
		this._id = id;
		_requiredVariables = new HashMap<String, String>();
	}
	
	/**
	 * <p>Returns the target identifier of the rand statement.</p>
	 * 
	 * @return target identifier
	 */
	public DataIdentifier getIdentifier()
	{
		return _id;
	}
	
	/**
	 * <p>Sets a parameter of type long.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 * @throws ParseException if a wrong parameter is specified
	 */
	public void addLongParam(String paramName, long paramValue) throws ParseException
	{
		if(paramName.equalsIgnoreCase(Param.ROWS.getParamName()))
		{
			rows = paramValue;
			return;
		}
		if(paramName.equalsIgnoreCase(Param.COLS.getParamName()))
		{
			cols = paramValue;
			return;
		}
		if(paramName.equalsIgnoreCase(Param.MIN.getParamName()))
		{
			minValue = (double) paramValue;
			return;
		}
		if(paramName.equalsIgnoreCase(Param.MAX.getParamName()))
		{
			maxValue = (double) paramValue;
			return;
		}
		
		throw new ParseException("unexpected long parameter \"" + paramName + "\"");
	}
	
	/**
	 * <p>Sets a parameter of type double.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 * @throws ParseException if a wrong parameter is specified
	 */
	public void addDoubleParam(String paramName, double paramValue) throws ParseException
	{
		if(paramName.equalsIgnoreCase(Param.MIN.getParamName()))
		{
			minValue = paramValue;
			return;
		}
		if(paramName.equalsIgnoreCase(Param.MAX.getParamName()))
		{
			maxValue = paramValue;
			return;
		}
		if(paramName.equalsIgnoreCase(Param.SPARSITY.getParamName()))
		{
			sparsity = paramValue;
			return;
		}
		
		throw new ParseException("unexpected double parameter \"" + paramName + "\"");
	}
	
	/**
	 * <p>Sets a parameter of type string.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 * @throws ParseException if a wrong parameter is specified
	 */
	public void addStringParam(String paramName, String paramValue) throws ParseException
	{
		if(paramName.equalsIgnoreCase(Param.ROWS.getParamName()))
		{
			rows = Long.parseLong(paramValue);
			return;
		}
		if(paramName.equalsIgnoreCase(Param.COLS.getParamName()))
		{
			cols = Long.parseLong(paramValue);
			return;
		}
		if(paramName.equalsIgnoreCase(Param.MIN.getParamName()))
		{
			minValue = Double.parseDouble(paramValue);
			return;
		}
		if(paramName.equalsIgnoreCase(Param.MAX.getParamName()))
		{
			maxValue = Double.parseDouble(paramValue);
			return;
		}
		if(paramName.equalsIgnoreCase(Param.SPARSITY.getParamName()))
		{
			sparsity = Double.parseDouble(paramValue);
			return;
		}
		if(paramName.equalsIgnoreCase(Param.PDF.getParamName()))
		{
		    probabilityDensityFunction = paramValue;
		    return;
		}
		
		throw new ParseException("unexpected string parameter \"" + paramName + "\"");
	}
	
	/**
	 * <p>Sets a parameter of type variable.</p>
	 * 
	 * @param paramName parameter name
	 * @param paramValue parameter value
	 */
	public void addVarParam(String paramName, DataIdentifier paramValue)
	{
	    _requiredVariables.put(paramName, paramValue.getName());
	}
	
	/**
	 * <p>Validates the spcified parameters.</p>
	 * 
	 * @throws ParseException if a parameter is defined wrong
	 */
	public void validateFunctionCall() throws ParseException
	{
		if(rows < 1)
			throw new ParseException("the number of rows needs to be 1 or larger");
		if(cols < 1)
			throw new ParseException("the number of columns needs to be 1 or larger");
		if(sparsity < 0 || sparsity > 1)
			throw new ParseException("the sparsity has to be between 0 and 1");
	}
	
	/**
	 * <p>Sets the properties of the target identifier according to the specified parameters.</p>
	 * @throws LanguageException 
	 */
	public void setIdentifierProperties() throws ParseException
	{
		_id.setFormatType(FormatType.BINARY);
		_id.setValueType(ValueType.DOUBLE);
		_id.setDimensions(rows, cols);
		_id.computeDataType();
	}
	
	/**
	 * <p>Returns the number of rows of the random object.</p>
	 * 
	 * @return number of rows
	 */
	public long getRows()
	{
		return rows;
	}
	
	/**
	 * <p>Returns the number of columns of the random object.</p>
	 * 
	 * @return number of columns
	 */
	public long getCols()
	{
		return cols;
	}
	
	/**
	 * <p>Returns the minimum random value.</p>
	 * 
	 * @return minimum value
	 */
	public double getMinValue()
	{
		return minValue;
	}
	
	/**
	 * <p>Returns the maximum random value.</p>
	 * 
	 * @return maximum value
	 */
	public double getMaxValue()
	{
		return maxValue;
	}
	
	/**
	 * <p>Returns the sparsity of the random object.</p>
	 * 
	 * @return sparsity
	 */
	public double getSparsity()
	{
		return sparsity;
	}
	
	/**
	 * <p>Returns the probability densitfy function used to generate the random object.</p>
	 * 
	 * @return probability density function
	 */
	public String getProbabilityDensityFunction()
	{
		return probabilityDensityFunction;
	}
	
	/**
	 * <p>Returns all variables which are required.</p>
	 * 
	 * @return required variables
	 */
	public String[] getRequiredVariables()
	{
	    ArrayList<String> variables = new ArrayList<String>();
	    for(String variable : _requiredVariables.values())
	    {
	        if(!variables.contains(variable))
	            variables.add(variable);
	    }
	    
	    return variables.toArray(new String[variables.size()]);
	}
	
	/**
	 * <p>Updates the required variables with their value.</p>
	 * 
	 * @param variables list of variables
	 * @throws ParseException if parameter is not available or value of a variable is not allowed
	 * @throws LanguageException 
	 */
	public void updateVariables(HashMap<String, ConstIdentifier> variables) throws ParseException
	{
	    for(String paramName : _requiredVariables.keySet())
	    {
	    	String varToReplace = _requiredVariables.get(paramName);
	        if(!variables.containsKey(varToReplace))
	            throw new ParseException("variable " + varToReplace + " is not available");
	        String paramValue = null;
	        if(variables.get(varToReplace) instanceof IntIdentifier)
	            paramValue = Long.toString(((IntIdentifier) variables.get(varToReplace)).getValue());
	        else if(variables.get(varToReplace) instanceof DoubleIdentifier)
                paramValue = Double.toString(((DoubleIdentifier) variables.get(varToReplace)).getValue());
	        addStringParam(paramName, paramValue);
	    }
	    setIdentifierProperties();
	}
	
	@Override
	public boolean controlStatement()
	{
		return false;
	}

	@Override
	public VariableSet initializebackwardLV(VariableSet lo)
	{
		return lo;
	}

	@Override
	public void initializeforwardLV(VariableSet activeIn)
	{
		
	}

	@Override
	public VariableSet variablesRead()
	{
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated()
	{
		VariableSet result = new VariableSet();
		result.addVariable(_id.getName(), _id);
		return result;
	}
    
    /**
     * <p>Returns a string representation of the rand function call.</p>
     */
    public String toString()
    {
        StringBuffer sb = new StringBuffer();
        sb.append(_id.getName() + " = Rand(");
        sb.append("rows=" + rows);
        sb.append(", cols=" + cols);
        sb.append(", min=" + minValue);
        sb.append(", max=" + maxValue);
        sb.append(", sparsity=" + sparsity);
        sb.append(", pdf=" + probabilityDensityFunction);
        
        sb.append(");");
        return sb.toString();
    }
}
