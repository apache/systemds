/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.parser;

import java.util.ArrayList;
import java.util.HashMap;
import java.io.IOException;


public class FunctionCallIdentifier extends DataIdentifier 
{
	
	private ArrayList<ParameterExpression> _paramExprs;
	private FunctCallOp _opcode;	// stores whether internal or external
	private String _namespace;		// namespace of the function being called (null if current namespace is to be used)

	/**
	 * setFunctionName: sets the function namespace (if specified) and name
	 * @param functionName the (optional) namespace information and name of function.  If both namespace and name are specified, they are concatinated with "::"
	 * @throws DMLParseException 
	 */
	public void setFunctionName(String functionName) throws DMLParseException{
		_name = functionName;
	}
	
	public void setFunctionNamespace(String passed) throws DMLParseException{
		_namespace 	= passed;
	}
	
	public String getNamespace(){
		return _namespace;
	}
	
	public ArrayList<ParameterExpression> getParamExprs(){
		return _paramExprs;
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException {
			
		ArrayList<ParameterExpression> newParameterExpressions = new ArrayList<ParameterExpression>();
		for (ParameterExpression paramExpr : _paramExprs)
			newParameterExpressions.add(new ParameterExpression(paramExpr.getName(), paramExpr.getExpr().rewriteExpression(prefix)));
		
		// rewrite each output expression
		FunctionCallIdentifier fci = new FunctionCallIdentifier(newParameterExpressions);
		
		fci.setBeginLine(this.getBeginLine());
		fci.setBeginColumn(this.getBeginColumn());
		fci.setEndLine(this.getEndLine());
		fci.setEndColumn(this.getEndColumn());
			
		fci._name = this._name;
		fci._namespace = this._namespace;
		fci._opcode = this._opcode;
		fci._kind = Kind.FunctionCallOp;	 
		
		return fci;
	}
	
	
	
	public FunctionCallIdentifier(){}
	
	public FunctionCallIdentifier(ArrayList<ParameterExpression> paramExpressions) {
		
		_paramExprs = paramExpressions;
		_opcode = null;
		_kind = Kind.FunctionCallOp;	 
	}
	
	
	
	public FunctCallOp getOpCode() {
		return _opcode;
	}
	
	/**
	 * Validate parse tree : Process ExtBuiltinFunction Expression is an
	 * assignment statement
	 * 
	 * NOTE: this does not override the normal validateExpression because it needs to pass dmlp!
	 * 
	 * @throws LanguageException
	 */
	public void validateExpression(DMLProgram dmlp, HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> constVars, boolean conditional) 
		throws LanguageException, IOException
	{
		// Step 1: check the namespace exists, and that function is defined in the namespace
		if (dmlp.getNamespaces().get(_namespace) == null){
			raiseValidateError("namespace " + _namespace + " is not defined ", conditional);
		}
		FunctionStatementBlock fblock = dmlp.getFunctionStatementBlock(_namespace, _name);
		if (fblock == null){
			raiseValidateError("function " + _name + " is undefined in namespace " + _namespace, conditional);
		}
		
		// Step 2: set opcode (whether internal or external function) -- based on whether FunctionStatement
		// in FunctionStatementBlock is ExternalFunctionStatement or FunctionStatement
		if (fblock.getStatement(0) instanceof ExternalFunctionStatement)
			_opcode = Expression.FunctCallOp.EXTERNAL;
		else
			_opcode = Expression.FunctCallOp.INTERNAL;
		
		// Step 3: check all parameters to be either unnamed or named for functions
		boolean hasNamed = false, hasUnnamed = false;
		for( ParameterExpression paramExpr : _paramExprs ) {
			if (paramExpr.getName() == null)
				hasUnnamed = true;
			else
				hasNamed = true;
		}
		if (hasNamed && hasUnnamed){
			raiseValidateError(" In DML, functions can only have named parameters " +
					"(e.g., name1=value1, name2=value2) or unnamed parameters (e.g, value1, value2). " + 
					_name + " has both parameter types.", conditional);
		}
		
		// Step 4: validate expressions for each passed parameter
		for( ParameterExpression paramExpr : _paramExprs ) {
			if (paramExpr.getExpr() instanceof FunctionCallIdentifier) {
				raiseValidateError("UDF function call not supported as parameter to function call", false);
			}
			paramExpr.getExpr().validateExpression(ids, constVars, conditional);
		}
		
		// Step 5: constant propagation into function call statement
		for( ParameterExpression paramExpr : _paramExprs ) {
			Expression expri = paramExpr.getExpr();
			if( expri instanceof DataIdentifier && !(expri instanceof IndexedIdentifier)
				&& constVars.containsKey(((DataIdentifier)expri).getName()) )
			{
				//replace varname with constant in function call expression
				paramExpr.setExpr(constVars.get(((DataIdentifier)expri).getName()));
			}
		}
	
		// Step 6: check correctness of number of arguments and their types 
		FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
		if (fstmt.getInputParams().size() < _paramExprs.size()) { 
			raiseValidateError("function " + _name 
					+ " has incorrect number of parameters. Function requires " 
					+ fstmt.getInputParams().size() + " but was called with " + _paramExprs.size(), conditional);
		}
		
		// Step 7: set the outputs for the function
		_outputs = new Identifier[fstmt.getOutputParams().size()];
		for(int i=0; i < fstmt.getOutputParams().size(); i++) {
			_outputs[i] = new DataIdentifier(fstmt.getOutputParams().get(i));
		}
		
		return;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		if (_namespace != null && _namespace.length() > 0 && !_namespace.equals(".defaultNS")) 
			sb.append(_namespace + "::"); 
		sb.append(_name);
		sb.append(" ( ");		
				
		for (int i = 0; i < _paramExprs.size(); i++){
			sb.append(_paramExprs.get(i).toString());
			if (i<_paramExprs.size() - 1) 
				sb.append(",");
		}
		sb.append(" )");
		return sb.toString();
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		for (int i = 0; i < _paramExprs.size(); i++)
			result.addVariables(_paramExprs.get(i).getExpr().variablesRead());
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		for (int i=0; i< _outputs.length; i++)
			result.addVariable( ((DataIdentifier)_outputs[i]).getName(), (DataIdentifier)_outputs[i] );
		return result;
	}

	@Override
	public boolean multipleReturns() {
		return true;
	}
}


