package dml.parser;

import java.util.HashMap;
import java.util.Map;

import dml.parser.Expression.DataType;
import dml.utils.LanguageException;

public class ParameterizedBuiltinFunctionExpression extends Expression {

	private ParameterizedBuiltinFunctionOp _opcode;
	private HashMap<String,Expression> _varParams;
	
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
		return new ParameterizedBuiltinFunctionExpression(_opcode, _varParams );
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
	public void validateExpression(HashMap<String, DataIdentifier> ids)
			throws LanguageException {
		
		// validate all input parameters
		for ( String s : getVarParams().keySet() ) {
			getVarParam(s).validateExpression(ids);
			
			if ( getVarParam(s).getOutput().getDataType() != DataType.SCALAR ) {
			//	throw new LanguageException("Non-scalar data types are not supported for parameterized builtin functions.", LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
		
		String outputName = getTempName();
		DataIdentifier output = new DataIdentifier(outputName);
		//output.setProperties(this.getFirstExpr().getOutput());
		this.setOutput(output);

		// IMPORTANT: for each operation, one must handle unnamed parameters
		
		switch (this.getOpCode()) {
		
		case GROUPEDAGG:
			
			if (getVarParam("target")  == null || getVarParam("groups") == null)
				throw new LanguageException("Must define both target and groups and both must have same dimensions");
			
			if (getVarParam("target") instanceof DataIdentifier && getVarParam("groups") instanceof DataIdentifier && (getVarParam("weights") == null || getVarParam("weights") instanceof DataIdentifier))
			{
				
				DataIdentifier targetid = (DataIdentifier)getVarParam("target");
				DataIdentifier groupsid = (DataIdentifier)getVarParam("groups");
				DataIdentifier weightsid = (DataIdentifier)getVarParam("weights");
				
			
				if (targetid.getDim1() != groupsid.getDim1() || targetid.getDim2() != groupsid.getDim2() )
					throw new LanguageException("target and groups must have same dimensions");
				
				if (weightsid != null && (targetid.getDim1() != weightsid.getDim1() || targetid.getDim2() != weightsid.getDim2() ))
					throw new LanguageException("target and weights must have same dimensions");
			}
			
			
			if (getVarParam("fn") == null)
					throw new LanguageException("must define function name (fname=<function name>) for groupedAggregate()");
				 
			
			Expression functParam = getVarParam("fn");
			
			if (functParam instanceof Identifier){
			
				// standardize to lowercase and dequote fname
				String fnameStr = getVarParam("fn").toString();
				
				
				// check that IF fname="centralmoment" THEN order=m is defined, where m=2,3,4 
				// check ELSE IF fname is allowed
				if(fnameStr.equals("centralmoment")){
					String orderStr = getVarParam("order") == null ? null : getVarParam("order").toString();
					if (orderStr == null || !(orderStr.equals("2") || orderStr.equals("3") || orderStr.equals("4")))
						throw new LanguageException("for centralmoment, must define order.  Order must be equal to 2,3, or 4");
				}	
				else if (fnameStr.equals("count") 
						|| fnameStr.equals("sum") 
						|| fnameStr.equals("mean")
						|| fnameStr.equals("variance")){}
				else
					throw new LanguageException("fname is " + fnameStr + " but must be either centeralmoment, count, sum, mean, variance");
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
				throw new LanguageException("Quantile to cumulativeProbability() must be a scalar value.",
						LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			// Output is a scalar
			output.setDataType(DataType.SCALAR);
			output.setValueType(ValueType.DOUBLE);
			output.setDimensions(0, 0);

			break;

		default:
			throw new LanguageException("Unsupported parameterized function "
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
