package com.ibm.bi.dml.parser;

import java.util.HashMap;
import com.ibm.bi.dml.utils.LanguageException;


public class ParameterizedBuiltinFunctionExpression extends DataIdentifier {

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
			
			if (getVarParam("target")  == null || getVarParam("groups") == null)
				throw new LanguageException(this.printErrorLocation() + "Must define both target and groups and both must have same dimensions");
			
			if (getVarParam("target") instanceof DataIdentifier && getVarParam("groups") instanceof DataIdentifier && (getVarParam("weights") == null || getVarParam("weights") instanceof DataIdentifier))
			{
				
				DataIdentifier targetid = (DataIdentifier)getVarParam("target");
				DataIdentifier groupsid = (DataIdentifier)getVarParam("groups");
				DataIdentifier weightsid = (DataIdentifier)getVarParam("weights");
				
			
				if (targetid.getDim1() != groupsid.getDim1() || targetid.getDim2() != groupsid.getDim2() ){
					throw new LanguageException(this.printErrorLocation() + "target and groups must have same dimensions -- " 
							+ " targetid dims: " + targetid.getDim1() +" rows, " + targetid.getDim2() + " cols -- groupsid dims: " + groupsid.getDim1() + " rows, " + groupsid.getDim2() + " cols " );
				}
				if (weightsid != null && (targetid.getDim1() != weightsid.getDim1() || targetid.getDim2() != weightsid.getDim2() )){
					throw new LanguageException(this.printErrorLocation() + "target and weights must have same dimensions -- "
							+ " targetid dims: " + targetid.getDim1() +" rows, " + targetid.getDim2() + " cols -- weightsid dims: " + weightsid.getDim1() + " rows, " + weightsid.getDim2() + " cols " );
				}
			}
			
			
			if (getVarParam("fn") == null)
					throw new LanguageException(this.printErrorLocation() + "must define function name (fname=<function name>) for groupedAggregate()");
				 
			
			Expression functParam = getVarParam("fn");
			
			if (functParam instanceof Identifier){
			
				// standardize to lowercase and dequote fname
				String fnameStr = getVarParam("fn").toString();
				
				
				// check that IF fname="centralmoment" THEN order=m is defined, where m=2,3,4 
				// check ELSE IF fname is allowed
				if(fnameStr.equals("centralmoment")){
					String orderStr = getVarParam("order") == null ? null : getVarParam("order").toString();
					if (orderStr == null || !(orderStr.equals("2") || orderStr.equals("3") || orderStr.equals("4")))
						throw new LanguageException(this.printErrorLocation() + "for centralmoment, must define order.  Order must be equal to 2,3, or 4");
				}	
				else if (fnameStr.equals("count") 
						|| fnameStr.equals("sum") 
						|| fnameStr.equals("mean")
						|| fnameStr.equals("variance")){}
				else
					throw new LanguageException(this.printErrorLocation() + "fname is " + fnameStr + " but must be either centeralmoment, count, sum, mean, variance");
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

}
