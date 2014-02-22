/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.HashMap;


public class ParameterizedBuiltinFunctionExpression extends DataIdentifier 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ParameterizedBuiltinFunctionOp _opcode;
	private HashMap<String,Expression> _varParams;
	
	
	
	public static ParameterizedBuiltinFunctionExpression getParamBuiltinFunctionExpression(String functionName, HashMap<String,Expression> varParams){
	
		// check if the function name is built-in function
		//	 (assign built-in function op if function is built-in
		Expression.ParameterizedBuiltinFunctionOp pbifop = null;	
		if (functionName.equals("cumulativeProbability"))
			pbifop = Expression.ParameterizedBuiltinFunctionOp.CDF;
		else if (functionName.equals("groupedAggregate"))
			pbifop = Expression.ParameterizedBuiltinFunctionOp.GROUPEDAGG;
		else if (functionName.equals("removeEmpty"))
			pbifop = Expression.ParameterizedBuiltinFunctionOp.RMEMPTY;
		else if (functionName.equals("replace"))
			pbifop = Expression.ParameterizedBuiltinFunctionOp.REPLACE;
		else
			return null;
		
		ParameterizedBuiltinFunctionExpression retVal = new ParameterizedBuiltinFunctionExpression(pbifop,varParams);
		return retVal;
	} // end method getBuiltinFunctionExpression
			
	public ParameterizedBuiltinFunctionExpression(ParameterizedBuiltinFunctionOp op, HashMap<String,Expression> varParams) {
		_kind = Kind.ParameterizedBuiltinFunctionOp;
		_opcode = op;
		_varParams = varParams;
	}

	public ParameterizedBuiltinFunctionExpression() {
		_kind = Kind.ParameterizedBuiltinFunctionOp;
		_opcode = ParameterizedBuiltinFunctionOp.INVALID;
		_varParams = new HashMap<String,Expression>();
	}

	public Expression rewriteExpression(String prefix) throws LanguageException {
		
		HashMap<String,Expression> newVarParams = new HashMap<String,Expression>();
		for (String key : _varParams.keySet()){
			Expression newExpr = _varParams.get(key).rewriteExpression(prefix);
			newVarParams.put(key, newExpr);
		}	
		ParameterizedBuiltinFunctionExpression retVal = new ParameterizedBuiltinFunctionExpression(_opcode, newVarParams);
	
		retVal._beginLine 	= this._beginLine;
		retVal._beginColumn = this._beginColumn;
		retVal._endLine 	= this._endLine;
		retVal._endColumn	= this._endColumn;
	
		return retVal;
	}

	public void setOpcode(ParameterizedBuiltinFunctionOp op) {
		_opcode = op;
	}
	
	public ParameterizedBuiltinFunctionOp getOpCode() {
		return _opcode;
	}
	
	public HashMap<String,Expression> getVarParams() {
		return _varParams;
	}
	
	public Expression getVarParam(String name) {
		return _varParams.get(name);
	}

	public void addVarParam(String name, Expression value){
		_varParams.put(name, value);
	}
	
	public void removeVarParam(String name) {
		_varParams.remove(name);
	}
	
	/**
	 * Validate parse tree : Process BuiltinFunction Expression in an assignment
	 * statement
	 * 
	 * @throws LanguageException
	 */
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars)
			throws LanguageException {
		
		// validate all input parameters
		for ( String s : getVarParams().keySet() ) {
			getVarParam(s).validateExpression(ids, constVars);
		}
		
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		//output.setProperties(this.getFirstExpr().getOutput());
		this.setOutput(output);

		// IMPORTANT: for each operation, one must handle unnamed parameters
		
		switch (this.getOpCode()) {
		
		case GROUPEDAGG:
			
			if (getVarParam(Statement.GAGG_TARGET)  == null || getVarParam(Statement.GAGG_GROUPS) == null){
				throw new LanguageException(this.printErrorLocation() + "Must define both target and groups and both must have same dimensions");
			}
			if (getVarParam(Statement.GAGG_TARGET) instanceof DataIdentifier && getVarParam(Statement.GAGG_GROUPS) instanceof DataIdentifier && (getVarParam(Statement.GAGG_WEIGHTS) == null || getVarParam(Statement.GAGG_WEIGHTS) instanceof DataIdentifier))
			{
				
				DataIdentifier targetid = (DataIdentifier)getVarParam(Statement.GAGG_TARGET);
				DataIdentifier groupsid = (DataIdentifier)getVarParam(Statement.GAGG_GROUPS);
				DataIdentifier weightsid = (DataIdentifier)getVarParam(Statement.GAGG_WEIGHTS);
				
				if( targetid.dimsKnown() && groupsid.dimsKnown() &&
					(targetid.getDim1() != groupsid.getDim1() || targetid.getDim2() != groupsid.getDim2()) )
				{
					
					throw new LanguageException(this.printErrorLocation() + "target and groups must have same dimensions -- " 
							+ " targetid dims: " + targetid.getDim1() +" rows, " + targetid.getDim2() + " cols -- groupsid dims: " + groupsid.getDim1() + " rows, " + groupsid.getDim2() + " cols " );
				}
				if( weightsid != null && (targetid.dimsKnown() && weightsid.dimsKnown()) &&
					(targetid.getDim1() != weightsid.getDim1() || targetid.getDim2() != weightsid.getDim2() ))
				{
					
					throw new LanguageException(this.printErrorLocation() + "target and weights must have same dimensions -- "
							+ " targetid dims: " + targetid.getDim1() +" rows, " + targetid.getDim2() + " cols -- weightsid dims: " + weightsid.getDim1() + " rows, " + weightsid.getDim2() + " cols " );
				}
			}
			
			
			if (getVarParam(Statement.GAGG_FN) == null){
				throw new LanguageException(this.printErrorLocation() + "must define function name (fname=<function name>) for groupedAggregate()");
			}
			
			Expression functParam = getVarParam(Statement.GAGG_FN);
			
			if (functParam instanceof Identifier){
			
				// standardize to lowercase and dequote fname
				String fnameStr = getVarParam(Statement.GAGG_FN).toString();
				
				
				// check that IF fname="centralmoment" THEN order=m is defined, where m=2,3,4 
				// check ELSE IF fname is allowed
				if(fnameStr.equals(Statement.GAGG_FN_CM)){
					String orderStr = getVarParam(Statement.GAGG_FN_CM_ORDER) == null ? null : getVarParam(Statement.GAGG_FN_CM_ORDER).toString();
					if (orderStr == null || !(orderStr.equals("2") || orderStr.equals("3") || orderStr.equals("4"))){
						throw new LanguageException(this.printErrorLocation() + "for centralmoment, must define order.  Order must be equal to 2,3, or 4");
					}
				}
				else if (fnameStr.equals(Statement.GAGG_FN_COUNT) 
						|| fnameStr.equals(Statement.GAGG_FN_SUM) 
						|| fnameStr.equals(Statement.GAGG_FN_MEAN)
						|| fnameStr.equals(Statement.GAGG_FN_VARIANCE)){}
				else {
					throw new LanguageException(this.printErrorLocation() + "fname is " + fnameStr + " but must be either centeralmoment, count, sum, mean, variance");
				}
			}
			
			// Output is a matrix with unknown dims
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(-1, -1);
				
			break; 
			
		case CDF:
			/*
			 * Usage: p = cumulativeProbability(x, dist="chisq", df=20);
			 */
			
			// CDF expects one unnamed parameter
			// it must be renamed as "quantile" 
			// (i.e., we must compute P(X <= x) where x is called as "quantile" )
			
			// check if quantile is of type SCALAR
			if ( getVarParam("target").getOutput().getDataType() != DataType.SCALAR ) {
				
				throw new LanguageException(this.printErrorLocation() + "Quantile to cumulativeProbability() must be a scalar value.",
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			// Output is a scalar
			output.setDataType(DataType.SCALAR);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(0, 0);

			break;
			
		case RMEMPTY:
		{
			//check existence and correctness of arguments
			Expression target = getVarParam("target");
			if( target==null ) {
				
				throw new LanguageException(this.printErrorLocation() + "Named parameter 'target' missing. Please specify the input matrix.",
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if( target.getOutput().getDataType() != DataType.MATRIX ){
				
				throw new LanguageException(this.printErrorLocation() + "Input matrix 'target' is of type '"+target.getOutput().getDataType()+"'. Please specify the input matrix.",
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);				
			}			
			Expression margin = getVarParam("margin");
			if( margin==null )
			{
				throw new LanguageException(this.printErrorLocation() + "Named parameter 'margin' missing. Please specify 'rows' or 'cols'.",
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if( !margin.toString().equals("rows") && !margin.toString().equals("cols") ){
				throw new LanguageException(this.printErrorLocation() + "Named parameter 'margin' has an invalid value '"+margin.toString()+"'. Please specify 'rows' or 'cols'.",
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);				
			}
			
			// Output is a matrix with unknown dims
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(-1, -1);
			
			break;
		}
		
		case REPLACE:
		{
			//check existence and correctness of arguments
			Expression target = getVarParam("target");
			if( target==null ) {				
				String error = this.printErrorLocation() + "Named parameter 'target' missing. Please specify the input matrix.";
				LOG.error( error );
				throw new LanguageException(error, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if( target.getOutput().getDataType() != DataType.MATRIX ){
				String error = this.printErrorLocation() + "Input matrix 'target' is of type '"+target.getOutput().getDataType()+"'. Please specify the input matrix.";
				LOG.error( error );	
				throw new LanguageException(error, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);				
			}	
			
			Expression pattern = getVarParam("pattern");
			if( pattern==null ) {
				String error = this.printErrorLocation() + "Named parameter 'pattern' missing. Please specify the replacement pattern.";
				LOG.error( error );
				throw new LanguageException(error, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if( pattern.getOutput().getDataType() != DataType.SCALAR ){				
				String error = this.printErrorLocation() + "Replacement pattern 'pattern' is of type '"+pattern.getOutput().getDataType()+"'. Please, specify a scalar replacement pattern.";
				LOG.error( error );	
				throw new LanguageException(error, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);				
			}	
			
			Expression replacement = getVarParam("replacement");
			if( replacement==null ) {
				String error = this.printErrorLocation() + "Named parameter 'replacement' missing. Please specify the replacement value.";
				LOG.error( error );
				throw new LanguageException(error, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			else if( replacement.getOutput().getDataType() != DataType.SCALAR ){				
				String error = this.printErrorLocation() + "Replacement value 'replacement' is of type '"+replacement.getOutput().getDataType()+"'. Please, specify a scalar replacement value.";
				LOG.error( error );	
				throw new LanguageException(error, LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);				
			}	
			
			// Output is a matrix with same dims as input
			output.setDataType(DataType.MATRIX);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(target.getOutput().getDim1(), target.getOutput().getDim2());
			
			break;
		}

			
		default:
			
			throw new LanguageException(this.printErrorLocation() + "Unsupported parameterized function "
						+ this.getOpCode(),
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
		}
		return;
	}

	public String toString() {
		StringBuffer sb = new StringBuffer(_opcode.toString() + "(");

		 for (String key : _varParams.keySet()){
			 sb.append("," + key + "=" + _varParams.get(key));
		 }
		sb.append(" )");
		return sb.toString();
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		for (String s : _varParams.keySet()) {
			result.addVariables ( _varParams.get(s).variablesRead() );
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		for (String s : _varParams.keySet()) {
			result.addVariables ( _varParams.get(s).variablesUpdated() );
		}
		result.addVariable(((DataIdentifier)this.getOutput()).getName(), (DataIdentifier)this.getOutput());
		return result;
	}

	@Override
	public boolean multipleReturns() {
		return false;
	}
}
