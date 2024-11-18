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

public class StringIdentifier extends ConstIdentifier 
{
	private String _val;

	@Override
	public Expression rewriteExpression(String prefix) {
		return this;
	}

	public StringIdentifier(String val, ParseInfo parseInfo) {
		super();
		setInfo(val);
		setParseInfo(parseInfo);
	}

	public StringIdentifier(ParserRuleContext ctx, String val, String filename) {
		super();
		setInfo(val);
		setCtxValuesAndFilename(ctx, filename);
	}

	private void setInfo(String val) {
		_val = val;
		setDimensions(0, 0);
		computeDataType();
		setValueType(ValueType.STRING);
	}

	public String getValue(){
		return _val;
	}

	public void setValue(String val) { _val = val; }
	
	@Override
	public String toString(){
		return _val;
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
	public long getLongValue() {
		throw new LanguageException("Unsupported string-to-long conversion.");
	}
}
