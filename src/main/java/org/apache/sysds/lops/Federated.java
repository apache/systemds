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

package org.apache.sysds.lops;


import java.util.HashMap;

import static org.apache.sysds.common.Types.DataType;
import static org.apache.sysds.common.Types.ValueType;
import static org.apache.sysds.parser.DataExpression.FED_ADDRESSES;
import static org.apache.sysds.parser.DataExpression.FED_FTYPE;
import static org.apache.sysds.parser.DataExpression.FED_LOCAL_OBJECTS;
import static org.apache.sysds.parser.DataExpression.FED_RANGES;
import static org.apache.sysds.parser.DataExpression.FED_TYPE;

public class Federated extends Lop {
	private Lop _type, _addresses, _ranges, _objects, _fType;
	
	public Federated(HashMap<String, Lop> inputLops, DataType dataType, ValueType valueType) {
		super(Type.Federated, dataType, valueType);
		_type = inputLops.get(FED_TYPE);
		_addresses = inputLops.get(FED_ADDRESSES);

		if(inputLops.size() == 4) {
			_objects = inputLops.get(FED_LOCAL_OBJECTS);
			addInput(_objects);
			_fType = inputLops.get(FED_FTYPE);
			addInput(_fType);
		}
		else {
			_ranges = inputLops.get(FED_RANGES);
			addInput(_ranges);
			_ranges.addOutput(this);
		}

		addInput(_type);
		_type.addOutput(this);
		addInput(_addresses);
		_addresses.addOutput(this);
	}
	
	@Override
	public String getInstructions(String type, String addresses, String ranges, String output) {
		StringBuilder sb = new StringBuilder("FED");
		sb.append(OPERAND_DELIMITOR);
		sb.append("fedinit");
		sb.append(OPERAND_DELIMITOR);
		sb.append(_type.prepScalarInputOperand(type));
		sb.append(OPERAND_DELIMITOR);
		sb.append(_addresses.prepScalarInputOperand(addresses));
		sb.append(OPERAND_DELIMITOR);
		sb.append(_ranges.prepScalarInputOperand(ranges));
		sb.append(OPERAND_DELIMITOR);
		sb.append(prepOutputOperand(output));
		return sb.toString();
	}

	@Override
	public String getInstructions(String objects, String fType, String type, String addresses, String output) {
		StringBuilder sb = new StringBuilder("FED");
		sb.append(OPERAND_DELIMITOR);
		sb.append("fedinit");
		sb.append(OPERAND_DELIMITOR);
		sb.append(_type.prepScalarInputOperand(type));
		sb.append(OPERAND_DELIMITOR);
		sb.append(_addresses.prepScalarInputOperand(addresses));
		sb.append(OPERAND_DELIMITOR);
		sb.append(_objects.prepScalarInputOperand(objects));
		sb.append(OPERAND_DELIMITOR);
		sb.append(_fType.prepScalarInputOperand(fType));
		sb.append(OPERAND_DELIMITOR);
		sb.append(prepOutputOperand(output));
		return sb.toString();
	}
	
	@Override
	public String toString() {
		return "FedInit";
	}
}
