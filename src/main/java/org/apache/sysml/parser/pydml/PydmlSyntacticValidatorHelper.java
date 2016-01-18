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

package org.apache.sysml.parser.pydml;

import java.util.ArrayList;
import java.util.List;

import org.antlr.v4.runtime.Token;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.pydml.PydmlParser.FunctionCallAssignmentStatementContext;
import org.apache.sysml.parser.pydml.PydmlParser.ParameterizedExpressionContext;
import org.apache.sysml.parser.pydml.PydmlSyntacticErrorListener.CustomDmlErrorListener;


public class PydmlSyntacticValidatorHelper 
{	
	private CustomDmlErrorListener _errorListener = null;
	public PydmlSyntacticValidatorHelper(CustomDmlErrorListener errorListener) {
		this._errorListener = errorListener;
	}
	public void notifyErrorListeners(String message, int line, int charPositionInLine) {
		this._errorListener.validationError(line, charPositionInLine, message);
	}
	
	public void notifyErrorListeners(String message, Token op) {
		this._errorListener.validationError(op.getLine(), op.getCharPositionInLine(), message);
	}

	public void raiseWarning(String message, Token op) {
		this._errorListener.validationWarning(op.getLine(), op.getCharPositionInLine(), message);
	}
	
	public String getCurrentFileName() {
		return _errorListener.peekFileName();
	}
	
	// Returns list of two elements <namespace, function names>, else null
	public ArrayList<String> getQualifiedNames(String fullyQualifiedFunctionName) {
		String [] fnNames = fullyQualifiedFunctionName.split("\\."); // instead of ::
		String functionName = "";
		String namespace = "";
		if(fnNames.length == 1) {
			namespace = DMLProgram.DEFAULT_NAMESPACE;
			functionName = fnNames[0].trim();
		}
		else if(fnNames.length == 2) {
			namespace = fnNames[0].trim();
			functionName = fnNames[1].trim();
		}
		else
			return null;
		
		ArrayList<String> retVal = new ArrayList<String>();
		retVal.add(namespace);
		retVal.add(functionName);
		return retVal;
	}
	
	public boolean validateBuiltinFunctions(FunctionCallAssignmentStatementContext ctx) {
		String functionName = ctx.name.getText().replaceAll(" ", "").trim();
		if(functionName.compareTo("write") == 0 || functionName.compareTo(DMLProgram.DEFAULT_NAMESPACE + ".write") == 0) {
			return validateBuiltinWriteFunction(ctx);
		}
		return true;
	}
	
	private boolean validateBuiltinWriteFunction(FunctionCallAssignmentStatementContext ctx) {
		
		return true;
	}
	
	public ArrayList<org.apache.sysml.parser.ParameterExpression> getParameterExpressionList(List<ParameterizedExpressionContext> paramExprs) {
		ArrayList<org.apache.sysml.parser.ParameterExpression> retVal = new ArrayList<org.apache.sysml.parser.ParameterExpression>();
		for(ParameterizedExpressionContext ctx : paramExprs) {
			String paramName = null;
			if(ctx.paramName != null && ctx.paramName.getText() != null && !ctx.paramName.getText().isEmpty()) {
				paramName = ctx.paramName.getText();
			}
			org.apache.sysml.parser.ParameterExpression myArg = new org.apache.sysml.parser.ParameterExpression(paramName, ctx.paramVal.info.expr);
			retVal.add(myArg);
		}
		return retVal;
	}
}
