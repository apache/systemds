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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;
import org.apache.sysml.runtime.util.DataConverter;

// TODO rename to MatrixScalar...
public class ScalarMatrixArithmeticCPInstruction extends ArithmeticBinaryCPInstruction
{
	
	public ScalarMatrixArithmeticCPInstruction(Operator op, 
											   CPOperand in1, 
											   CPOperand in2, 
											   CPOperand out, 
											   String opcode,
											   String istr){
		super(op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) throws DMLRuntimeException {
		CPOperand mat = (input1.getDataType() == DataType.MATRIX) ? input1 : input2;
		CPOperand scalar = (input1.getDataType() == DataType.MATRIX) ? input2 : input1;

		ValueType svt = scalar.getValueType();
		ValueType mvt = mat.getValueType();

		if (mvt == ValueType.STRING) { // print("this " + X[1,1] + " that");
			ScalarObject matObj = (ScalarObject) ec.getScalarInput(mat.getName(), mat.getValueType(), mat.isLiteral());
			ScalarObject scalarObj = (ScalarObject) ec.getScalarInput(scalar.getName(), scalar.getValueType(),
					scalar.isLiteral());
			boolean matrixThenScalar = false;
			if (input1.getDataType() == DataType.MATRIX) {
				matrixThenScalar = true;
			}
			String outString = null;
			if (matrixThenScalar) {
				outString = matObj.toString() + scalarObj.toString();
			} else {
				outString = scalarObj.toString() + matObj.toString();
			}
			StringObject so = new StringObject(outString);
			ec.setScalarOutput(output.getName(), so);
		} else if (svt == ValueType.STRING) { // print("this " + X[1,1]); or
												// print(X[1,1] + " that");
			MatrixBlock inBlock = ec.getMatrixInput(mat.getName());
			if ((inBlock.getNumRows() == 1) && (inBlock.getNumColumns() == 1)) {
				DoubleObject doubleObject = new DoubleObject(inBlock.getValue(0, 0));

				ScalarObject scalarObj = (ScalarObject) ec.getScalarInput(scalar.getName(), scalar.getValueType(),
						scalar.isLiteral());

				boolean matrixThenScalar = false;
				if (input1.getDataType() == DataType.MATRIX) {
					matrixThenScalar = true;
				}

				String outString = null;
				if (matrixThenScalar) {
					outString = doubleObject.toString() + scalarObj.toString();
				} else {
					outString = scalarObj.toString() + doubleObject.toString();
				}
				StringObject so = new StringObject(outString);

				ec.releaseMatrixInput(mat.getName());
				ec.setScalarOutput(output.getName(), so);
			} else {
				String matrixString = DataConverter.toString(inBlock);
				ec.releaseMatrixInput(mat.getName());

				ScalarObject scalarObj = (ScalarObject) ec.getScalarInput(scalar.getName(), scalar.getValueType(),
						scalar.isLiteral());

				boolean matrixThenScalar = false;
				if (input1.getDataType() == DataType.MATRIX) {
					matrixThenScalar = true;
				}

				String outString = null;
				if (matrixThenScalar) {
					outString = matrixString + scalarObj.toString();
				} else {
					outString = scalarObj.toString() + matrixString;
				}
				StringObject so = new StringObject(outString);

				ec.setScalarOutput(output.getName(), so);
			}
		} else {
			MatrixBlock inBlock = ec.getMatrixInput(mat.getName());
			ScalarObject constant = (ScalarObject) ec.getScalarInput(scalar.getName(), scalar.getValueType(),
					scalar.isLiteral());

			ScalarOperator sc_op = (ScalarOperator) _optr;
			sc_op.setConstant(constant.getDoubleValue());

			MatrixBlock retBlock = (MatrixBlock) inBlock.scalarOperations(sc_op, new MatrixBlock());

			ec.releaseMatrixInput(mat.getName());

			// Ensure right dense/sparse output representation (guarded by
			// released input memory)
			if (checkGuardedRepresentationChange(inBlock, retBlock)) {
				retBlock.examSparsity();
			}

			ec.setMatrixOutput(output.getName(), retBlock);
		}
	}
}
