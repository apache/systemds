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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;

public class ListIdentifier extends Identifier {

	public ListIdentifier(){
		_dim1 = -1;
		_dim2 = -1;
		_dataType = DataType.LIST;
		_valueType = ValueType.UNKNOWN;
		_blocksize = -1;
		_nnz = -1;
		setOutput(this);
		_format = null;
	}

	@Override
	public Expression rewriteExpression(String prefix) {
		throw new UnsupportedOperationException("Unimplemented method 'rewriteExpression'");
	}

	@Override
	public VariableSet variablesRead() {
		throw new UnsupportedOperationException("Unimplemented method 'variablesRead'");
	}

	@Override
	public VariableSet variablesUpdated() {
		throw new UnsupportedOperationException("Unimplemented method 'variablesUpdated'");
	}
	
}
