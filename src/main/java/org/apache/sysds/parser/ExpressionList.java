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
import java.util.HashMap;

public class ExpressionList extends Expression {

	protected String _name;
	protected ArrayList<Expression> _value;

	public ExpressionList(ArrayList<Expression> value) {
		this._name = "tmp";
		this._value = value;
	}

	public String getName() {
		return _name;
	}

	public void setName(String _name) {
		this._name = _name;
	}

	public ArrayList<Expression> getValue() {
		return _value;
	}

	public void setValue(ArrayList<Expression> _value) {
		this._value = _value;
	}

	public Identifier getOutput() {
		return new ListIdentifier();
	}

	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars,
		boolean conditional) {
		for(Expression ex : _value) {
			ex.validateExpression(ids, currConstVars, conditional);
		}
	}

	@Override
	public Expression rewriteExpression(String prefix) {
		throw new LanguageException("ExpressionList should not be exposed beyond parser layer.");
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		for(Expression expr : _value) {
			result.addVariables(expr.variablesRead());
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		for(Expression expr : _value) {
			result.addVariables(expr.variablesUpdated());
		}
		return result;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append("[");
		for(Expression e : _value) {
			sb.append(e);
		}
		sb.append("]");
		return sb.toString();
	}
}
