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

package org.apache.sysds.parser;

import java.util.ArrayList;
import java.util.List;

import org.antlr.v4.runtime.ParserRuleContext;
import org.apache.sysds.common.Opcodes;

public class PrintStatement extends Statement
{
	/**
	 * The PRINTTYPE options are: PRINT, PRINTF, and STOP.
	 * <p>
	 * Note that PRINTF functionality is overloaded onto the existing 'print'
	 * built-in function.
	 */
	public enum PRINTTYPE {
		PRINT, PRINTF, STOP, ASSERT
	}

	protected PRINTTYPE _type; // print, printf, or stop
	protected List<Expression> expressions;

	private static PRINTTYPE getPrintType(String type, List<Expression> expressions) {
		if(type.equalsIgnoreCase(Opcodes.PRINT.toString())) {
			if ((expressions == null) || (expressions.size() == 1)) {
				return PRINTTYPE.PRINT;
			} else {
				return PRINTTYPE.PRINTF;
			}
		}
		else if (type.equalsIgnoreCase(Opcodes.ASSERT.toString())) {
			return PRINTTYPE.ASSERT;
		}
		else if (type.equalsIgnoreCase(Opcodes.STOP.toString())) {
			return PRINTTYPE.STOP;
		}
		else
			throw new LanguageException("Unknown statement type: " + type);
	}

	public PrintStatement(ParserRuleContext ctx, String type, String filename) {
		this(getPrintType(type, null), null);
		setCtxValues(ctx);
		setFilename(filename);
	}

	public PrintStatement(ParserRuleContext ctx, String type, List<Expression> expressions, String filename) {
		this(getPrintType(type, expressions), expressions);
		setCtxValues(ctx);
		setFilename(filename);
	}

	public PrintStatement(PRINTTYPE type, List<Expression> expressions) {
		_type = type;
		if (expressions == null) {
			this.expressions = new ArrayList<>();
		} else {
			this.expressions = expressions;
		}
	}

	@Override
	public Statement rewriteStatement(String prefix) {
		List<Expression> newExpressions = new ArrayList<>();
		for (Expression oldExpression : expressions) {
			Expression newExpression = oldExpression.rewriteExpression(prefix);
			newExpressions.add(newExpression);
		}
		PrintStatement retVal = new PrintStatement(_type, newExpressions);
		retVal.setParseInfo(this);
		return retVal;
	}
	
	@Override
	public void initializeforwardLV(VariableSet activeIn){}
	
	@Override
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(_type + "(");
		if ((_type == PRINTTYPE.PRINT) || (_type == PRINTTYPE.STOP) || (_type == PRINTTYPE.ASSERT)) {
			Expression expression = expressions.get(0);
			if (expression instanceof StringIdentifier) {
				sb.append("\"");
				sb.append(expression.toString());
				sb.append("\"");
			} else {
				sb.append(expression.toString());
			}
		} else if (_type == PRINTTYPE.PRINTF) {
			for (int i = 0; i < expressions.size(); i++) {
				if (i > 0) {
					sb.append(", ");
				}
				Expression expression = expressions.get(i);
				if (expression instanceof StringIdentifier) {
					sb.append("\"");
					sb.append(expression.toString());
					sb.append("\"");
				} else {
					sb.append(expression.toString());
				}
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
		// Keep stop() statement in a separate statement block
		return (getType() == PRINTTYPE.STOP);
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
