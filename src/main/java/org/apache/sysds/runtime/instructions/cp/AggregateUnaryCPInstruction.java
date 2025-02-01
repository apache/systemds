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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.data.BasicTensorBlock;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageDedupUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.LibMatrixCountDistinct;
import org.apache.sysds.runtime.matrix.data.LibMatrixSketch;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;
import org.apache.sysds.runtime.matrix.operators.UnarySketchOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.utils.Explain;

public class AggregateUnaryCPInstruction extends UnaryCPInstruction {
	protected static final Log LOG = LogFactory.getLog(AggregateUnaryCPInstruction.class.getName());

	public enum AUType {
		NROW, NCOL, LENGTH, EXISTS, LINEAGE, 
		COUNT_DISTINCT, COUNT_DISTINCT_APPROX, UNIQUE,
		DEFAULT;
		public boolean isMeta() {
			return this != DEFAULT;
		}
	}
	
	private final AUType _type;
	
	private AggregateUnaryCPInstruction(Operator op, CPOperand in, CPOperand out, AUType type, String opcode, String istr) {
		this(op, in, null, null, out, type, opcode, istr);
	}

	protected AggregateUnaryCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			AUType type, String opcode, String istr) {
		super(CPType.AggregateUnary, op, in1, in2, in3, out, opcode, istr);
		_type = type;
	}
	
	public static AggregateUnaryCPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		
		if(opcode.equalsIgnoreCase(Opcodes.NROW.toString()) || opcode.equalsIgnoreCase(Opcodes.NCOL.toString())
			|| opcode.equalsIgnoreCase(Opcodes.LENGTH.toString()) || opcode.equalsIgnoreCase(Opcodes.EXISTS.toString())
			|| opcode.equalsIgnoreCase(Opcodes.LINEAGE.toString())){
			return new AggregateUnaryCPInstruction(new SimpleOperator(Builtin.getBuiltinFnObject(opcode)),
				in1, out, AUType.valueOf(opcode.toUpperCase()), opcode, str);
		} 
		else if(opcode.equalsIgnoreCase(Opcodes.UACD.toString())
				|| opcode.equalsIgnoreCase(Opcodes.UACDR.toString())
				|| opcode.equalsIgnoreCase(Opcodes.UACDC.toString())){
			AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode,
					Integer.parseInt(parts[3]));
			return new AggregateUnaryCPInstruction(aggun, in1, out, AUType.COUNT_DISTINCT, opcode, str);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.UACDAP.toString())
				|| opcode.equalsIgnoreCase(Opcodes.UACDAPR.toString())
				|| opcode.equalsIgnoreCase(Opcodes.UACDAPC.toString())){
			AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode,
					Integer.parseInt(parts[3]));
			return new AggregateUnaryCPInstruction(aggun, in1, out, AUType.COUNT_DISTINCT_APPROX,
					opcode, str);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.UARIMAX.toString()) || opcode.equalsIgnoreCase(Opcodes.UARIMIN.toString())){
			// parse with number of outputs
			AggregateUnaryOperator aggun = InstructionUtils
				.parseAggregateUnaryRowIndexOperator(opcode, Integer.parseInt(parts[4]), Integer.parseInt(parts[3]));
			return new AggregateUnaryCPInstruction(aggun, in1, out, AUType.DEFAULT, opcode, str);
		}
		else if(opcode.equalsIgnoreCase(Opcodes.UNIQUE.toString())
				|| opcode.equalsIgnoreCase(Opcodes.UNIQUER.toString())
				|| opcode.equalsIgnoreCase(Opcodes.UNIQUEC.toString())){
			AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode,
					Integer.parseInt(parts[3]));
			return new AggregateUnaryCPInstruction(aggun, in1, out, AUType.UNIQUE, opcode, str);
		}
		else { //DEFAULT BEHAVIOR
			AggregateUnaryOperator aggun = InstructionUtils
				.parseBasicAggregateUnaryOperator(opcode, Integer.parseInt(parts[3]));
			return new AggregateUnaryCPInstruction(aggun, in1, out, AUType.DEFAULT, opcode, str);
		}
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		String outputName = output.getName();
		String opcode = getOpcode();

		switch( _type ) {
			case NROW:
			case NCOL:
			case LENGTH: {
				//check existence of input variable
				if( !ec.getVariables().keySet().contains(input1.getName()) )
					throw new DMLRuntimeException("Variable '"+input1.getName()+"' does not exist.");
				
				//get meta data information
				long rval = -1;
				if (input1.getDataType() == DataType.LIST && _type == AUType.LENGTH ) {
					rval = ((ListObject)ec.getVariable(input1.getName())).getLength();
				}
				else if( input1.getDataType().isMatrix() || input1.getDataType().isFrame() ) {
					DataCharacteristics mc = ec.getDataCharacteristics(input1.getName());
					rval = getSizeMetaData(_type, mc);
		
					//check for valid output, and acquire read if necessary
					//(Use case: In case of forced exec type singlenode, there are no reblocks. For csv
					//we however, support unspecified input sizes, which requires a read to obtain the
					//required meta data)
					//Note: check on matrix characteristics to cover incorrect length (-1*-1 -> 1)
					if( !mc.dimsKnown() ) //invalid nrow/ncol/length
					{
						if( DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE 
							|| input1.getDataType() == DataType.FRAME )
						{
							//read the input matrix/frame and explicitly refresh meta data
							CacheableData<?> obj = ec.getCacheableData(input1.getName());
							obj.acquireRead();
							obj.refreshMetaData();
							obj.release();
							
							//update meta data information
							mc = ec.getDataCharacteristics(input1.getName());
							rval = getSizeMetaData(_type, mc);
						}
						else {
							throw new DMLRuntimeException("Invalid meta data returned by '"+opcode+"': "+rval + ":" + instString);
						}
					}
				}
				
				//create and set output scalar
				ec.setScalarOutput(outputName, new IntObject(rval));
				break;
			}
			case EXISTS: {
				//probe existence of variable in symbol table w/o error
				String varName = !input1.isScalar() ? input1.getName() :
					ec.getScalarInput(input1).getStringValue();
				boolean rval = ec.getVariables().keySet().contains(varName);
				//create and set output scalar
				ec.setScalarOutput(outputName, new BooleanObject(rval));
				break;
			}
			case LINEAGE: {
				//serialize lineage and set output scalar
				if( ec.getLineageItem(input1) == null )
					throw new DMLRuntimeException("Lineage trace "
						+ "for variable "+input1.getName()+" unavailable.");
				
				LineageItem li = ec.getLineageItem(input1);
				String out = !DMLScript.LINEAGE_DEDUP ? Explain.explain(li) :
					Explain.explain(li) + "\n" + LineageDedupUtils.mergeExplainDedupBlocks(ec);
				ec.setScalarOutput(outputName, new StringObject(out));
				break;
			}
			case COUNT_DISTINCT:
			case COUNT_DISTINCT_APPROX: {
				if(!ec.getVariables().keySet().contains(input1.getName())) {
					throw new DMLRuntimeException("Variable '" + input1.getName() + "' does not exist.");
				}
				MatrixBlock input = ec.getMatrixInput(input1.getName());

				// Operator type: test and cast
				if (!(_optr instanceof CountDistinctOperator)) {
					throw new DMLRuntimeException("Operator should be instance of " + CountDistinctOperator.class.getSimpleName());
				}
				CountDistinctOperator op = (CountDistinctOperator) _optr;

				if (op.getDirection().isRowCol()) {
					long res = (long) LibMatrixCountDistinct.estimateDistinctValues(input, op).get(0, 0);
					ec.releaseMatrixInput(input1.getName());
					ec.setScalarOutput(outputName, new IntObject(res));
				} else {  // Row/Col
					// Note that for each row, the max number of distinct values < NNZ < max number of columns = 1000:
					// Since count distinct approximate estimates are unreliable for values < 1024,
					// we will force a naive count.
					MatrixBlock res = LibMatrixCountDistinct.estimateDistinctValues(input, op);
					ec.releaseMatrixInput(input1.getName());
					ec.setMatrixOutput(outputName, res);
				}

				break;
			}

			case UNIQUE: {
				if(!ec.getVariables().keySet().contains(input1.getName())) {
					throw new DMLRuntimeException("Variable '" + input1.getName() + "' does not exist.");
				}
				MatrixBlock input = ec.getMatrixInput(input1.getName());

				// Operator type: test and cast
				if (!(_optr instanceof UnarySketchOperator)) {
					throw new DMLRuntimeException("Operator should be instance of "
							+ UnarySketchOperator.class.getSimpleName());
				}
				UnarySketchOperator op = (UnarySketchOperator) _optr;

				MatrixBlock res = LibMatrixSketch.getUniqueValues(input, op.getDirection());
				ec.releaseMatrixInput(input1.getName());
				ec.setMatrixOutput(outputName, res);
				break;
			}

			default: {
				AggregateUnaryOperator au_op = (AggregateUnaryOperator) _optr;
				if (input1.getDataType() == DataType.MATRIX) {
					MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
					
					MatrixBlock resultBlock = matBlock.aggregateUnaryOperations(au_op, new MatrixBlock(),
						matBlock.getNumRows(), new MatrixIndexes(1, 1), true);

					ec.releaseMatrixInput(input1.getName());
					if (output.getDataType() == DataType.SCALAR) {
						DoubleObject ret = new DoubleObject(resultBlock.get(0, 0));
						ec.setScalarOutput(outputName, ret);
					} else {
						// since the computed value is a scalar, allocate a "temp" output matrix
						ec.setMatrixOutput(outputName, resultBlock);
					}
				} 
				else if (input1.getDataType() == DataType.TENSOR) {
					// TODO support DataTensor
					BasicTensorBlock basicTensor = ec.getTensorInput(input1.getName()).getBasicTensor();

					BasicTensorBlock resultBlock = basicTensor.aggregateUnaryOperations(au_op, new BasicTensorBlock());

					ec.releaseTensorInput(input1.getName());
					if(output.getDataType() == DataType.SCALAR)
						ec.setScalarOutput(outputName, ScalarObjectFactory.createScalarObject(
							input1.getValueType(), resultBlock.get(new int[]{0, 0})));
					else
						ec.setTensorOutput(outputName, new TensorBlock(resultBlock));
				}
				else {
					throw new DMLRuntimeException(opcode + " only supported on matrix or tensor.");
				}
			}
		}
	}
	
	public AUType getAUType() {
		return _type;
	}
	
	private static long getSizeMetaData(AUType type, DataCharacteristics mc) {
		switch( type ) {
			case NROW: return mc.getRows();
			case NCOL: return mc.getCols();
			case LENGTH: return mc.getRows() * mc.getCols();
			default:
				throw new RuntimeException("Opcode not applicable: "+type.name());
		}
	}
}
