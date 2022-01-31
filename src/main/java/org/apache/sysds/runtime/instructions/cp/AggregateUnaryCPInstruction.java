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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.data.BasicTensorBlock;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageDedupUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.LibMatrixCountDistinct;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.utils.Explain;

public class AggregateUnaryCPInstruction extends UnaryCPInstruction {
	// private static final Log LOG = LogFactory.getLog(AggregateUnaryCPInstruction.class.getName());

	public enum AUType {
		NROW, NCOL, LENGTH, EXISTS, LINEAGE, 
		COUNT_DISTINCT, COUNT_DISTINCT_APPROX,
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
		
		if(opcode.equalsIgnoreCase("nrow") || opcode.equalsIgnoreCase("ncol") 
			|| opcode.equalsIgnoreCase("length") || opcode.equalsIgnoreCase("exists")
			|| opcode.equalsIgnoreCase("lineage")){
			return new AggregateUnaryCPInstruction(new SimpleOperator(Builtin.getBuiltinFnObject(opcode)),
				in1, out, AUType.valueOf(opcode.toUpperCase()), opcode, str);
		} 
		else if(opcode.equalsIgnoreCase("uacd")){
			return new AggregateUnaryCPInstruction(new SimpleOperator(null),
			in1, out, AUType.COUNT_DISTINCT, opcode, str);
		}
		else if(opcode.equalsIgnoreCase("uacdap")){
			CountDistinctOperator op = new CountDistinctOperator(AUType.COUNT_DISTINCT_APPROX)
					.setDirection(Types.Direction.RowCol)
					.setIndexFunction(ReduceAll.getReduceAllFnObject());

			return new AggregateUnaryCPInstruction(op, in1, out, AUType.COUNT_DISTINCT_APPROX,
					opcode, str);
		}
		else if(opcode.equalsIgnoreCase("uacdapr")){
			CountDistinctOperator op = new CountDistinctOperator(AUType.COUNT_DISTINCT_APPROX)
					.setDirection(Types.Direction.Row)
					.setIndexFunction(ReduceCol.getReduceColFnObject());

			return new AggregateUnaryCPInstruction(op, in1, out, AUType.COUNT_DISTINCT_APPROX,
					opcode, str);
		}
		else if(opcode.equalsIgnoreCase("uacdapc")){
			CountDistinctOperator op = new CountDistinctOperator(AUType.COUNT_DISTINCT_APPROX)
					.setDirection(Types.Direction.Col)
					.setIndexFunction(ReduceRow.getReduceRowFnObject());

			return new AggregateUnaryCPInstruction(op, in1, out, AUType.COUNT_DISTINCT_APPROX,
					opcode, str);
		}
		else if(opcode.equalsIgnoreCase("uarimax") || opcode.equalsIgnoreCase("uarimin")){
			// parse with number of outputs
			AggregateUnaryOperator aggun = InstructionUtils
				.parseAggregateUnaryRowIndexOperator(opcode, Integer.parseInt(parts[4]), Integer.parseInt(parts[3]));
			return new AggregateUnaryCPInstruction(aggun, in1, out, AUType.DEFAULT, opcode, str);
		}
		else { //DEFAULT BEHAVIOR
			AggregateUnaryOperator aggun = InstructionUtils
				.parseBasicAggregateUnaryOperator(opcode, Integer.parseInt(parts[3]));
			return new AggregateUnaryCPInstruction(aggun, in1, out, AUType.DEFAULT, opcode, str);
		}
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		String output_name = output.getName();
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
				ec.setScalarOutput(output_name, new IntObject(rval));
				break;
			}
			case EXISTS: {
				//probe existence of variable in symbol table w/o error
				String varName = !input1.isScalar() ? input1.getName() :
					ec.getScalarInput(input1).getStringValue();
				boolean rval = ec.getVariables().keySet().contains(varName);
				//create and set output scalar
				ec.setScalarOutput(output_name, new BooleanObject(rval));
				break;
			}
			case LINEAGE: {
				//serialize lineage and set output scalar
				if( ec.getLineageItem(input1) == null )
					throw new DMLRuntimeException("Lineage trace "
						+ "for variable "+input1.getName()+" unavailable.");
				
				LineageItem li = ec.getLineageItem(input1);
				String out = !DMLScript.LINEAGE_DEDUP ? Explain.explain(li) :
					Explain.explain(li) + LineageDedupUtils.mergeExplainDedupBlocks(ec);
				ec.setScalarOutput(output_name, new StringObject(out));
				break;
			}
			case COUNT_DISTINCT: {
				if( !ec.getVariables().keySet().contains(input1.getName()) )
					throw new DMLRuntimeException("Variable '" + input1.getName() + "' does not exist.");
				MatrixBlock input = ec.getMatrixInput(input1.getName());
				CountDistinctOperator op = new CountDistinctOperator(_type);
				int res = LibMatrixCountDistinct.estimateDistinctValues(input, op);
				ec.releaseMatrixInput(input1.getName());
				ec.setScalarOutput(output_name, new IntObject(res));
				break;
			}
			case COUNT_DISTINCT_APPROX: {
				if(!ec.getVariables().keySet().contains(input1.getName())) {
					throw new DMLRuntimeException("Variable '" + input1.getName() + "' does not exist.");
				}

				MatrixBlock input = ec.getMatrixInput(input1.getName());
				if (!(_optr instanceof CountDistinctOperator)) {
					throw new DMLRuntimeException("Operator should be instance of " + CountDistinctOperator.class.getSimpleName());
				}

				CountDistinctOperator op = (CountDistinctOperator) _optr;  // It is safe to cast at this point

				if (op.getDirection().isRowCol()) {
					int res = LibMatrixCountDistinct.estimateDistinctValues(input, op);
					ec.releaseMatrixInput(input1.getName());
					ec.setScalarOutput(output_name, new IntObject(res));

				} else if (op.getDirection().isRow()) {
					MatrixBlock res = input.slice(0, input.getNumRows() - 1, 0, 0);
					for (int i = 0; i < input.getNumRows(); ++i) {
						res.setValue(i, 0, LibMatrixCountDistinct.estimateDistinctValues(input.slice(i, i), op));
					}
					ec.releaseMatrixInput(input1.getName());
					ec.setMatrixOutput(output_name, res);

				} else if (op.getDirection().isCol()) {
					MatrixBlock res = input.slice(0, 0, 0, input.getNumColumns() - 1);
					for (int j = 0; j < input.getNumColumns(); ++j) {
						res.setValue(0, j, LibMatrixCountDistinct.estimateDistinctValues(input.slice(0, input.getNumRows() - 1, j, j), op));
					}
					ec.releaseMatrixInput(input1.getName());
					ec.setMatrixOutput(output_name, res);

				} else {
					throw new DMLRuntimeException("Direction for CountDistinctOperator not recognized");
				}

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
						DoubleObject ret = new DoubleObject(resultBlock.getValue(0, 0));
						ec.setScalarOutput(output_name, ret);
					} else {
						// since the computed value is a scalar, allocate a "temp" output matrix
						ec.setMatrixOutput(output_name, resultBlock);
					}
				} 
				else if (input1.getDataType() == DataType.TENSOR) {
					// TODO support DataTensor
					BasicTensorBlock basicTensor = ec.getTensorInput(input1.getName()).getBasicTensor();

					BasicTensorBlock resultBlock = basicTensor.aggregateUnaryOperations(au_op, new BasicTensorBlock());

					ec.releaseTensorInput(input1.getName());
					if(output.getDataType() == DataType.SCALAR)
						ec.setScalarOutput(output_name, ScalarObjectFactory.createScalarObject(
							input1.getValueType(), resultBlock.get(new int[]{0, 0})));
					else
						ec.setTensorOutput(output_name, new TensorBlock(resultBlock));
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
