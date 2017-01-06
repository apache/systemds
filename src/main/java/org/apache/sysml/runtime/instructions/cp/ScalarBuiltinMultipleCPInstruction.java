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

package org.apache.sysml.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.lops.MultipleCP;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.operators.Operator;

/**
 * The ScalarBuiltinMultipleCPInstruction class is responsible for printf-style
 * Java-based string formatting. The first input is the format string. The
 * inputs after the first input are the arguments to be formatted in the format
 * string.
 *
 */
public class ScalarBuiltinMultipleCPInstruction extends BuiltinMultipleCPInstruction {

	public ScalarBuiltinMultipleCPInstruction(Operator op, String opcode, String istr, CPOperand output,
			CPOperand... inputs) {
		super(op, opcode, istr, output, inputs);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		if (MultipleCP.OperationType.PRINTF.toString().equalsIgnoreCase(getOpcode())) {
			List<ScalarObject> scalarObjects = new ArrayList<ScalarObject>();
			for (CPOperand input : inputs) {
				ScalarObject so = ec.getScalarInput(input.getName(), input.getValueType(), input.isLiteral());
				scalarObjects.add(so);
			}

			// determine the format string (first argument) to pass to String.format
			ScalarObject formatStringObject = scalarObjects.get(0);
			if (formatStringObject.getValueType() != Expression.ValueType.STRING) {
				throw new DMLRuntimeException("First parameter needs to be a string");
			}
			String formatString = formatStringObject.getStringValue();

			// determine the arguments after the format string to pass to String.format
			Object[] objects = null;
			if (scalarObjects.size() > 1) {
				objects = new Object[scalarObjects.size() - 1];
				for (int i = 1; i < scalarObjects.size(); i++) {
					ScalarObject scalarObject = scalarObjects.get(i);
					switch (scalarObject.getValueType()) {
					case INT:
						objects[i - 1] = scalarObject.getLongValue();
						break;
					case DOUBLE:
						objects[i - 1] = scalarObject.getDoubleValue();
						break;
					case BOOLEAN:
						objects[i - 1] = scalarObject.getBooleanValue();
						break;
					case STRING:
						objects[i - 1] = scalarObject.getStringValue();
						break;
					default:
					}
				}
			}

			String result = String.format(formatString, objects);
			if (!DMLScript.suppressPrint2Stdout()) {
				System.out.println(result);
			}

			// this is necessary so that the remove variable operation can be
			// performed
			ec.setScalarOutput(output.getName(), new StringObject(result));
		} else {
			throw new DMLRuntimeException(
					"Opcode (" + getOpcode() + ") not recognized in ScalarBuiltinMultipleCPInstruction");
		}

	}

}
