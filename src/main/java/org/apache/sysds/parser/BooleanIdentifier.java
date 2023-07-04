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

import org.antlr.v4.runtime.ParserRuleContext;
import org.apache.sysds.common.Types.ValueType;

public class BooleanIdentifier extends ConstIdentifier 
{
	private boolean _val;

	public BooleanIdentifier(boolean val) {
		super();
		setInfo(val);
		setBeginLine(-1);
		setBeginColumn(-1);
		setEndLine(-1);
		setEndColumn(-1);
		setText(null);
	}

	public BooleanIdentifier(boolean val, ParseInfo parseInfo) {
		this(val);
		setParseInfo(parseInfo);
	}

	public BooleanIdentifier(ParserRuleContext ctx, boolean val, String filename) {
		this(val);
		setCtxValuesAndFilename(ctx, filename);
	}

	private void setInfo(boolean val) {
		_val = val;
		setDimensions(0, 0);
		computeDataType();
		setValueType(ValueType.BOOLEAN);
	}

	@Override
	public Expression rewriteExpression(String prefix) {
		return this;
	}
	
	public boolean getValue(){
		return _val;
	}
	
	@Override
	public String toString(){
		return Boolean.toString(_val);
	}
	
	@Override
	public VariableSet variablesRead() {
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
		return null;
	}
	
	@Override
	public long getLongValue(){
		return (getValue() ? 1 : 0);
	}
}
