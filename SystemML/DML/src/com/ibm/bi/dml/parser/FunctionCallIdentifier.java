package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;
import java.io.IOException;

import com.ibm.bi.dml.utils.LanguageException;

public class FunctionCallIdentifier extends DataIdentifier {

	private ArrayList<Expression> _inputParamExpressions;
	private ArrayList<DataIdentifier> _outputs;
	private FunctCallOp _opcode;	// stores whether internal or external
	private String _namespace;		// namespace of the function being called (null if current namespace is to be used)

	
	/**
	 * setFunctionName: sets the function namespace (if specified) and name
	 * @param functionName the (optional) namespace information and name of function.  If both namespace and name are specified, they are concatinated with "::"
	 * @throws ParseException 
	 */
	public void setFunctionName(String functionName) throws ParseException{
		_name = functionName;
	}
	
	public void setFunctionNamespace(String passed) throws ParseException{
		_namespace 	= passed;
	}
	
	public String getNamespace(){
		return _namespace;
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException {
		
		// rewrite each input expression
	
		ArrayList<Expression> newInputParamExpressions = new ArrayList<Expression>(); 
		for (Expression expr : _inputParamExpressions){
			Expression newExpr = expr.rewriteExpression(prefix);
			newInputParamExpressions.add(newExpr);
		}
			
		// rewrite each output expression
		FunctionCallIdentifier fci = new FunctionCallIdentifier(newInputParamExpressions);
		fci._name = this._name;
		fci._namespace = this._namespace;
		fci._opcode = this._opcode;
		return fci;
	}
	
	public FunctionCallIdentifier(ArrayList<Expression> paramExpressions) {
		_inputParamExpressions = paramExpressions;
		_opcode = null;
		_kind = Kind.FunctionCallOp;	 
	}
	
	public FunctCallOp getOpCode() {
		return _opcode;
	}

	public ArrayList<Expression> getParamExpressions(){
		return _inputParamExpressions;
	}
	
	/**
	 * Validate parse tree : Process ExtBuiltinFunction Expression is an
	 * assignment statement
	 * 
	 * @throws LanguageException
	 */
	public void validateExpression(DMLProgram dmlp, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException, IOException{
		
		// check the namespace exists, and that function is defined in the namespace
		if (dmlp.getNamespaces().get(_namespace) == null)
			throw new LanguageException("namespace " + _namespace + " is not defined ");
		
		FunctionStatementBlock fblock = dmlp.getFunctionStatementBlock(_namespace, _name);
		if (fblock == null){
			String printedNamespace = (_namespace == null) ? "current" : _namespace;
			throw new LanguageException("function " + _name + " is undefined in namespace " + printedNamespace );
		}
		// set opcode (whether internal or external function) -- based on whether FunctionStatement
		// in FunctionStatementBlock is ExternalFunctionStatement or FunctionStatement
		if (fblock.getStatement(0) instanceof ExternalFunctionStatement)
			_opcode = Expression.FunctCallOp.EXTERNAL;
		else
			_opcode = Expression.FunctCallOp.INTERNAL;
		
		// validate expressions for each passed parameter
		for (Expression cur : _inputParamExpressions) {
			cur.validateExpression(ids, constVars);
		}

		FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
		
		// check correctness of number of arguments and their types 
		if (fstmt.getInputParams().size() < _inputParamExpressions.size()){ 
			throw new LanguageException("function " + _name 
					+ " has incorrect number of parameters. Function requires " 
					+ fstmt.getInputParams().size() + " but was called with " + _inputParamExpressions.size());
		}
		
		// check the types of the input to see they match OR has default values
		for (int i = 0; i < fstmt.getInputParams().size(); i++) {
					
			if (i >= _inputParamExpressions.size()){
				// check a default value is provided for this variable
				if (fstmt.getInputParams().get(i).getDefaultValue() == null)
					throw new LanguageException("line " + fstmt.getInputParams().get(i).getDefinedLine() 
							+ ": parameter " + fstmt.getInputParams().get(i) + " must have default value");
			}
			
			else {
				Expression param = _inputParamExpressions.get(i);
				boolean sameDataType = param._output.getDataType().equals(fstmt.getInputParams().get(i).getDataType());
				if (!sameDataType)
					throw new LanguageException("parameter " + param.toString() + " does not have correct dataType");
				boolean sameValueType = param._output.getValueType().equals(fstmt.getInputParams().get(i).getValueType());
				if (!sameValueType)
					throw new LanguageException("parameter " + param.toString() + " does not have correct valueType");
			}
		}
	
		// set the outputs for the function
		_outputs = new ArrayList<DataIdentifier>();
		for (DataIdentifier outParam: fstmt.getOutputParams()){
			_outputs.add(new DataIdentifier(outParam));
		}
		
		return;
	}

	@Override
	public DataIdentifier getOutput() {
			
		return _outputs.get(0);
	}
	
	public ArrayList<DataIdentifier> getOutputs() {
		
		return _outputs;
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		if (_namespace != null && _namespace.length() > 0) 
			sb.append(_namespace + "::"); 
		sb.append(_name);
		sb.append(" ( ");		
				
		for (int i=0; i<_inputParamExpressions.size(); i++){
			sb.append(_inputParamExpressions.get(i).toString());
			if (i<_inputParamExpressions.size() - 1) sb.append(",");
		}
		sb.append(" )");
		return sb.toString();
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		for (int i=0; i<_inputParamExpressions.size(); i++)
			result.addVariables(_inputParamExpressions.get(i).variablesRead());
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		for (int i=0; i< _outputs.size(); i++)
			result.addVariable(_outputs.get(i).getName(), _outputs.get(i));
		return result;
	}

}
