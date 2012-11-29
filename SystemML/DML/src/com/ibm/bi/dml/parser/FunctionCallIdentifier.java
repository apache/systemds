package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;
import java.io.IOException;

import com.ibm.bi.dml.utils.LanguageException;

public class FunctionCallIdentifier extends DataIdentifier {

	private ArrayList<Expression> _inputParamExpressions;
	private HashMap<String,Expression> _namedInputParamExpressions;
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
			
		// rewrite each named input expression
		HashMap<String,Expression> newNamedInputParamExpressions = new HashMap<String,Expression>(); 
		for (String exprName : _namedInputParamExpressions.keySet()){
			Expression expr = _namedInputParamExpressions.get(exprName);
			Expression newExpr = expr.rewriteExpression(prefix);
			newNamedInputParamExpressions.put(exprName,newExpr);
		}
		
		
		// rewrite each output expression
		FunctionCallIdentifier fci = new FunctionCallIdentifier(newInputParamExpressions, newNamedInputParamExpressions);
		
		fci._beginLine 		= this._beginLine;
		fci._beginColumn 	= this._beginColumn;
		fci._endLine		= this._endLine;
		fci._endColumn		= this._endColumn;
			
		fci._name = this._name;
		fci._namespace = this._namespace;
		fci._opcode = this._opcode;
		fci._kind = Kind.FunctionCallOp;	 
		return fci;
	}
	
	public FunctionCallIdentifier(){}
	
	public FunctionCallIdentifier(ArrayList<Expression> paramExpressions, HashMap<String, Expression> namedParamExpresssions) {
		
		if (paramExpressions == null)
			_inputParamExpressions = new ArrayList<Expression>();
		_inputParamExpressions = paramExpressions;
		
		if (namedParamExpresssions == null)
			_namedInputParamExpressions = new HashMap<String, Expression>();
		_namedInputParamExpressions = namedParamExpresssions;
		
		_opcode = null;
		_kind = Kind.FunctionCallOp;	 
	}
	
	public FunctCallOp getOpCode() {
		return _opcode;
	}

	public ArrayList<Expression> getParamExpressions(){
		return _inputParamExpressions;
	}
	
	public HashMap<String,Expression> getNamedParamExpressions(){
		return _namedInputParamExpressions;
	}
	
	/**
	 * Validate parse tree : Process ExtBuiltinFunction Expression is an
	 * assignment statement
	 * 
	 * @throws LanguageException
	 */
	public void validateExpression(DMLProgram dmlp, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars) throws LanguageException, IOException{
		
		// check the namespace exists, and that function is defined in the namespace
		if (dmlp.getNamespaces().get(_namespace) == null){
			throw new LanguageException(this.printErrorLocation() + "namespace " + _namespace + " is not defined ");
		}
		FunctionStatementBlock fblock = dmlp.getFunctionStatementBlock(_namespace, _name);
		if (fblock == null){
			throw new LanguageException(this.printErrorLocation() + "function " + _name + " is undefined in namespace " + _namespace );
		}
		// set opcode (whether internal or external function) -- based on whether FunctionStatement
		// in FunctionStatementBlock is ExternalFunctionStatement or FunctionStatement
		if (fblock.getStatement(0) instanceof ExternalFunctionStatement)
			_opcode = Expression.FunctCallOp.EXTERNAL;
		else
			_opcode = Expression.FunctCallOp.INTERNAL;
		
		// force all parameters to be either unnammed or named
		if (_inputParamExpressions.size() > 0 && _namedInputParamExpressions.size() > 0){
			
			throw new LanguageException(this.printErrorLocation() + " In DML, functions can only have named parameters " +
						"(e.g., name1=value1, name2=value2) or unnamed parameters (e.g, value1, value2). " + 
						_name + " has both parameter types.");
		}
		// validate expressions for each passed parameter
		for (Expression curr : _inputParamExpressions) {
			curr.validateExpression(ids, constVars);
		}

		// validate expressions for each named passed parameter
		for (String key : _namedInputParamExpressions.keySet()){
			Expression curr = _namedInputParamExpressions.get(key);
			curr.validateExpression(ids, constVars);
		}
		
		FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
		
		// TODO: DRB: FIX THIS
		// check correctness of number of arguments and their types 
		if (fstmt.getInputParams().size() < _inputParamExpressions.size()){ 
			
			throw new LanguageException(this.printErrorLocation() + "function " + _name 
					+ " has incorrect number of parameters. Function requires " 
					+ fstmt.getInputParams().size() + " but was called with " + _inputParamExpressions.size());
		}
		
		// check the types of the input to see they match OR has default values
		for (int i = 0; i < fstmt.getInputParams().size(); i++) {
					
			if (i >= _inputParamExpressions.size()){
				// check a default value is provided for this variable
				if (fstmt.getInputParams().get(i).getDefaultValue() == null){
					throw new LanguageException(this.printErrorLocation() + "parameter " + fstmt.getInputParams().get(i) + " must have default value");
				}
			}
			
			else {
				Expression param = _inputParamExpressions.get(i);
				boolean sameDataType = param._output.getDataType().equals(fstmt.getInputParams().get(i).getDataType());
				if (!sameDataType){
					throw new LanguageException(this.printErrorLocation() + "parameter " + param.toString() + " does not have correct dataType");
				}
				boolean sameValueType = param._output.getValueType().equals(fstmt.getInputParams().get(i).getValueType());
				if (!sameValueType){
					throw new LanguageException(this.printErrorLocation() + "parameter " + param.toString() + " does not have correct valueType");
				}
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
