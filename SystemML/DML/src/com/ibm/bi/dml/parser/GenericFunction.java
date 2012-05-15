package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.lops.BuiltinFunction;
import com.ibm.bi.dml.lops.LopProperties;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.GenericFunctionParam.ParamType;


/**
 * <p>
 * Defines all built-in functions with their required and optional parameters.
 * </p>
 * 
 * @author schnetter
 * @author Felix Hamborg
 */
public enum GenericFunction {
	/**
	 * <p>
	 * Generates random data of the following kinds:
	 * </p>
	 * <ul>
	 * <li>Scalar</li>
	 * <li>Vector</li>
	 * <li>Matrix</li>
	 * </ul>
	 */
	RAND("OldRand") {
		@Override
		protected void initializeParamlist() {
			_requiredParams = new HashMap<String, ParamType[]>();
			_optionalParams = new HashMap<String, ParamType[]>();
			_optionalParams.put("rows", new ParamType[] { ParamType.INT, ParamType.STRING });
			_optionalParams.put("cols", new ParamType[] { ParamType.INT, ParamType.STRING });
			_optionalParams.put("min", new ParamType[] { ParamType.INT, ParamType.DOUBLE, ParamType.STRING });
			_optionalParams.put("max", new ParamType[] { ParamType.INT, ParamType.DOUBLE, ParamType.STRING });
			_optionalParams.put("sparsity", new ParamType[] { ParamType.INT, ParamType.DOUBLE, ParamType.STRING });
			_optionalParams.put("pdf", new ParamType[] { ParamType.STRING });
			_optionalParams.put("rows_in_block", new ParamType[] { ParamType.INT });
			_optionalParams.put("columns_in_block", new ParamType[] { ParamType.INT });
		}

		@Override
		public long getRowDimension(HashMap<String, GenericFunctionParam> params) {
			return getLongParamValue(params, "rows", 1);
		}

		@Override
		public long getColDimension(HashMap<String, GenericFunctionParam> params) {
			return getLongParamValue(params, "cols", 1);
		}

		public long getRowsInBlock(HashMap<String, GenericFunctionParam> params) {
			return getLongParamValue(params, "rows_in_block", 10000);
		}

		public long getColsInBlock(HashMap<String, GenericFunctionParam> params) {
			return getLongParamValue(params, "columns_in_block", 10000);
		}

		@Override
		public Lops getLop(String target,
				HashMap<String, GenericFunctionParam> params,
				ArrayList<Lops> inputs) {
			return new BuiltinFunction(target, params, inputs, this,
					getRowDimension(params), getColDimension(params), true, Format.BINARY, LopProperties.ExecLocation.MapOrReduce, DataType.UNKNOWN, ValueType.UNKNOWN);
		}
	};

	/** function name of built-in function */
	protected String _functionName;
	/** list of required parameters with their allowed parameter types */
	protected HashMap<String, ParamType[]> _requiredParams;
	/** list of optional parameters with their allowed parameter types */
	protected HashMap<String, ParamType[]> _optionalParams;

	/**
	 * <p>
	 * Creates a new built-in function and initializes the parameter list.
	 * </p>
	 * 
	 * @param functionName
	 *            function name
	 */
	private GenericFunction(String functionName) {
		_functionName = functionName;
		initializeParamlist();
	}

	/**
	 * <p>
	 * Returns the function name of the built-in function.
	 * </p>
	 * 
	 * @return function name
	 */
	public String getFunctionName() {
		return _functionName;
	}

	/**
	 * <p>
	 * Returns a list with required parameters and their allowed parameter types
	 * for the built-in function.
	 * </p>
	 * 
	 * @return list of requried parameters
	 */
	public HashMap<String, ParamType[]> getRequiredParams() {
		return _requiredParams;
	}

	/**
	 * <p>
	 * Checks if a parameter is in the list of possible parameters for the
	 * built-in function and if the parameter type is allowed for it.
	 * </p>
	 * <p>
	 * Returns the correct spelled parameter name or <b>null</b> if the
	 * parameter is not valid.
	 * </p>
	 * 
	 * @param param
	 *            parameter name
	 * @param paramType
	 *            parameter type
	 * @return correct spelled paramter name
	 */
	public String getParam(String param, ParamType paramType) {
		for (String key : _requiredParams.keySet()) {
			if (param.equalsIgnoreCase(key)) {
				for (ParamType type : _requiredParams.get(key)) {
					if (paramType == type)
						return key;
				}
				return null;
			}
		}

		for (String key : _optionalParams.keySet()) {
			if (param.equalsIgnoreCase(key)) {
				for (ParamType type : _optionalParams.get(key)) {
					if (paramType == type)
						return key;
				}
				return null;
			}
		}

		return null;
	}

	/**
	 * <p>
	 * Returns the long value of a parameter. If it is not specified the default
	 * value is returned instead.
	 * </p>
	 * 
	 * @param params
	 *            parameter list
	 * @param paramName
	 *            parameter name
	 * @param defaultValue
	 *            default value
	 * @return long value of the parameter or default value
	 */
	protected long getLongParamValue(HashMap<String, GenericFunctionParam> params, String paramName, long defaultValue) {
		if (params.containsKey(paramName)) {
			GenericFunctionParam param = params.get(paramName);
			switch (param.getParamType()) {
			case INT:
				return ((GenericFunctionLongParam) param).getParamValue();
			case STRING:
				return Long.parseLong(((GenericFunctionStringParam) param).getParamValue());
			}
		}

		return defaultValue;
	}

	/**
	 * <p>
	 * Returns the double value of a parameter. If it is not specified the
	 * default value is returned instead.
	 * </p>
	 * 
	 * @param params
	 *            parameter list
	 * @param paramName
	 *            parameter name
	 * @param defaultValue
	 *            default value
	 * @return double value of the parameter or default value
	 */
	protected double getDoubleParamValue(HashMap<String, GenericFunctionParam> params, String paramName,
			double defaultValue) {
		if (params.containsKey(paramName)) {
			GenericFunctionParam param = params.get(paramName);
			switch (param.getParamType()) {
			case INT:
				return (double) ((GenericFunctionLongParam) param).getParamValue();
			case DOUBLE:
				return ((GenericFunctionDoubleParam) param).getParamValue();
			case STRING:
				return Double.parseDouble(((GenericFunctionStringParam) param).getParamValue());
			}
		}

		return defaultValue;
	}

	/**
	 * <p>
	 * Initializes the parameter list for a built-in function.
	 * </p>
	 */
	protected abstract void initializeParamlist();

	/**
	 * <p>
	 * Calculates the row dimension of the built-in function's output.
	 * </p>
	 * 
	 * @param params
	 *            parameters which are passed to the built-in function
	 * @return number of rows
	 */
	public abstract long getRowDimension(HashMap<String, GenericFunctionParam> params);

	/**
	 * <p>
	 * Calculates the column dimension of the built-in function's output.
	 * </p>
	 * 
	 * @param params
	 *            parameters which are passed to the built-in function
	 * @return number of columns
	 */
	public abstract long getColDimension(HashMap<String, GenericFunctionParam> params);

	public abstract long getRowsInBlock(HashMap<String, GenericFunctionParam> params);

	public abstract long getColsInBlock(HashMap<String, GenericFunctionParam> params);

	/**
	 * <p>
	 * Creates a LOP for the built-in function.
	 * </p>
	 * 
	 * @param target
	 *            identifier where the output of the built-in function is stored
	 * @param params
	 *            parameters which are passed to the built-in function
	 * @return built-in function LOP
	 */
	public abstract Lops getLop(String target,
			HashMap<String, GenericFunctionParam> params,
			ArrayList<Lops> input);
}
