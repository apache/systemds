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
import java.util.List;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.debug.DMLBreakpointManager;

 
public class PrintStatement extends Statement
{
	/**
	 * The PRINTTYPE options are: PRINT, PRINTF, and STOP.
	 * <p>
	 * Note that PRINTF functionality is overloaded onto the existing 'print'
	 * built-in function.
	 */
	public enum PRINTTYPE {
		PRINT, PRINTF, STOP
	};

	protected PRINTTYPE _type; // print, printf, or stop
	protected List<Expression> expressions;

	private static PRINTTYPE getPrintType(String type, List<Expression> expressions) throws LanguageException {
		if(type.equalsIgnoreCase("print")) {
			if (expressions.size() == 1) {
				return PRINTTYPE.PRINT;
			} else {
				return PRINTTYPE.PRINTF;
			}
		}
		else if (type.equalsIgnoreCase("stop")) {
			return PRINTTYPE.STOP;
		}
		else
			throw new LanguageException("Unknown statement type: " + type);
	}

	public PrintStatement(String type, List<Expression> expressions, int beginLine, int beginCol,
			int endLine, int endCol) throws LanguageException {
		this(getPrintType(type, expressions), expressions);

		setBeginLine(beginLine);
		setBeginColumn(beginCol);
		setEndLine(endLine);
		setEndColumn(endCol);
	}

	public PrintStatement(PRINTTYPE type, List<Expression> expressions)
			throws LanguageException {
		_type = type;
		this.expressions = expressions;
	}

	public Statement rewriteStatement(String prefix) throws LanguageException{
		List<Expression> newExpressions = new ArrayList<Expression>();
		for (Expression oldExpression : expressions) {
			Expression newExpression = oldExpression.rewriteExpression(prefix);
			newExpressions.add(newExpression);
		}
		PrintStatement retVal = new PrintStatement(_type, newExpressions);
		retVal.setBeginLine(this.getBeginLine());
		retVal.setBeginColumn(this.getBeginColumn());
		retVal.setEndLine(this.getEndLine());
		retVal.setEndColumn(this.getEndColumn());
		
		return retVal;
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(_type + " (");
		if ((_type == PRINTTYPE.PRINT) || (_type == PRINTTYPE.STOP)) {
			sb.append(expressions.get(0).toString());
		} else if (_type == PRINTTYPE.PRINTF) {
			for (int i = 0; i < expressions.size(); i++) {
				if (i > 0) {
					sb.append(", ");
				}
				Expression expression = expressions.get(i);
				sb.append(expression.toString());
			}
		}

		sb.append(");");
		return sb.toString();
	}
	
	
	@Override
	public VariableSet variablesRead() {
		VariableSet variableSet = new VariableSet();
		for (Expression expression : expressions) {
			VariableSet variablesRead = expression.variablesRead();
			variableSet.addVariables(variablesRead);
		}
		return variableSet;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return new VariableSet();
	}

	@Override
	public boolean controlStatement() {	 
		// ensure that breakpoints end up in own statement block
		if (DMLScript.ENABLE_DEBUG_MODE) {
			DMLBreakpointManager.insertBreakpoint(expressions.get(0).getBeginLine());
			return true;
		}
		
		// Keep stop() statement in a separate statement block
		if(getType() == PRINTTYPE.STOP)
			return true;
		
		return false;
	}

	public PRINTTYPE getType() {
		return _type;
	}

	public List<Expression> getExpressions() {
		return expressions;
	}

	public void setExpressions(List<Expression> expressions) {
		this.expressions = expressions;
	}
}
