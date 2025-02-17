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

package org.apache.sysds.resource.cost;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.*;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

import static org.apache.sysds.resource.cost.IOCostUtils.IOMetrics;
import static org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;

public class CPCostUtils {
	private static final long DEFAULT_NFLOP_NOOP = 10;
	private static final long DEFAULT_NFLOP_CP = 1;
	private static final long DEFAULT_NFLOP_TEXT_IO = 350;
	private static final long DEFAULT_INFERRED_DIM = 1000000;

	public static double getVariableInstTime(VariableCPInstruction inst, VarStats input, VarStats output, IOMetrics metrics) {
		long nflop;
		switch (inst.getOpcode()) {
			case "write":
				String fmtStr = inst.getInput3().getLiteral().getStringValue();
				Types.FileFormat fmt = Types.FileFormat.safeValueOf(fmtStr);
				long xwrite = fmt.isTextFormat() ? DEFAULT_NFLOP_TEXT_IO : DEFAULT_NFLOP_CP;
				nflop = input.getCellsWithSparsity() * xwrite;
				break;
			case "cast_as_matrix":
			case "cast_as_frame":
				nflop = input.getCells();
				break;
			case "rmfilevar": case "attachfiletovar": case "setfilename":
				throw new RuntimeException("Undefined behaviour for instruction with opcode: " + inst.getOpcode());
			default:
				// negligibly low number of FLOP (independent of variables' dimensions)
				return 0;
		}
		// assignOutputMemoryStats() needed only for casts
		return getCPUTime(nflop, metrics, output, input);
	}

	public static double getDataGenCPInstTime(UnaryCPInstruction inst, VarStats output, IOMetrics metrics) {
		long nflop;
		String opcode = inst.getOpcode();
		if( inst instanceof DataGenCPInstruction) {
			if (opcode.equals("rand") || opcode.equals("frame")) {
				DataGenCPInstruction rinst = (DataGenCPInstruction) inst;
				if( rinst.getMinValue() == 0.0 && rinst.getMaxValue() == 0.0 )
					nflop = 0; // empty matrix
				else if( rinst.getSparsity() == 1.0 && rinst.getMinValue() == rinst.getMaxValue() ) // allocate, array fill
					nflop = 8 * output.getCells();
				else { // full rand
					if (rinst.getSparsity() == 1.0)
						nflop = 32 * output.getCells() + 8 * output.getCells(); // DENSE gen (incl allocate)
					else if (rinst.getSparsity() < MatrixBlock.SPARSITY_TURN_POINT)
						nflop = 3 * output.getCellsWithSparsity() + 24 * output.getCellsWithSparsity();  //SPARSE gen (incl allocate)
					else
						nflop = 2 * output.getCells() + 8 * output.getCells();  // DENSE gen (incl allocate)
				}
			} else if (opcode.equals(Opcodes.SEQUENCE.toString())) {
				nflop = DEFAULT_NFLOP_CP * output.getCells();
			} else {
				// DataGen.SAMPLE_OPCODE, DataGen.TIME_OPCODE,
				throw new RuntimeException("Undefined behaviour for instruction with opcode: " + inst.getOpcode());
			}
		}
		else if( inst instanceof StringInitCPInstruction) {
			nflop = DEFAULT_NFLOP_CP * output.getCells();
		} else {
			throw new IllegalArgumentException("Method has been called with invalid instruction: " + inst);
		}
		return getCPUTime(nflop, metrics, output);
	}

	public static double getUnaryInstTime(UnaryCPInstruction inst, VarStats input, VarStats weights, VarStats output, IOMetrics metrics) {
		if (inst instanceof UaggOuterChainCPInstruction || inst instanceof DnnCPInstruction) {
			throw new RuntimeException("Time estimation for CP instruction of class " + inst.getClass().getName() + "not supported yet");
		}
		// CPType = Unary/Builtin
		CPType instructionType = inst.getCPInstructionType();
		String opcode = inst.getOpcode();

		boolean includeWeights = false;
		if (inst instanceof MMTSJCPInstruction) {
			MMTSJ.MMTSJType type = ((MMTSJCPInstruction) inst).getMMTSJType();
			opcode += type.isLeft() ? "_left" : "_right";
		} else if (inst instanceof ReorgCPInstruction && opcode.equals(Opcodes.SORT.toString())) {
			if (inst.input2 != null) includeWeights = true;
		} else if (inst instanceof QuantileSortCPInstruction) {
			if (inst.input2 != null) {
				opcode += "_wts";
				includeWeights = true;
			}
		} else if (inst instanceof CentralMomentCPInstruction) {
			CMOperator.AggregateOperationTypes opType = ((CMOperator) inst.getOperator()).getAggOpType();
			opcode += "_" + opType.name().toLowerCase();
			if (inst.input2 != null) {
				includeWeights = true;
			}
		}
		long nflop = getInstNFLOP(instructionType, opcode, output, input);
		if (includeWeights)
			return getCPUTime(nflop, metrics, output, input, weights);
		if (!opcodeRequiresScan(opcode)) {
			return getCPUTime(nflop, metrics, output);
		}
		return getCPUTime(nflop, metrics, output, input);
	}

	public static double getBinaryInstTime(BinaryCPInstruction inst, VarStats input1, VarStats input2, VarStats weights, VarStats output, IOMetrics metrics) {
		// CPType = Binary/Builtin
		CPType instructionType = inst.getCPInstructionType();
		String opcode = inst.getOpcode();

		boolean includeWeights = false;
		if (inst instanceof CovarianceCPInstruction) { // cov
			includeWeights = true;
		} else if (inst instanceof QuantilePickCPInstruction) {
			PickByCount.OperationTypes opType = ((QuantilePickCPInstruction) inst).getOperationType();
			opcode += "_" + opType.name().toLowerCase();
		} else if (inst instanceof AggregateBinaryCPInstruction) {
			AggregateBinaryCPInstruction abinst = (AggregateBinaryCPInstruction) inst;
			opcode += abinst.transposeLeft? "_tl": "";
			opcode += abinst.transposeRight? "_tr": "";
		}
		long nflop = getInstNFLOP(instructionType, opcode, output, input1, input2);
		if (includeWeights)
			return getCPUTime(nflop, metrics, output, input1, input2, weights);
		return getCPUTime(nflop, metrics, output, input1, input2);
	}

	public static double getComputationInstTime(ComputationCPInstruction inst, VarStats input1, VarStats input2, VarStats input3, VarStats input4, VarStats output, IOMetrics metrics) {
		if (inst instanceof UnaryCPInstruction || inst instanceof BinaryCPInstruction) {
			throw new RuntimeException("Instructions of type UnaryCPInstruction and BinaryCPInstruction are not handled by this method");
		}
		CPType instructionType = inst.getCPInstructionType();
		String opcode = inst.getOpcode();

		// CURRENTLY: 2 is the maximum number of needed input stats objects for NFLOP estimation
		long nflop = getInstNFLOP(instructionType, opcode, output, input1, input2);
		return getCPUTime(nflop, metrics, output, input1, input2, input3, input4);
	}

	public static double getBuiltinNaryInstTime(BuiltinNaryCPInstruction inst, VarStats[] inputs, VarStats output, IOMetrics metrics) {
		CPType instructionType = inst.getCPInstructionType();
		String opcode = inst.getOpcode();
		long nflop;
		if (inputs == null) {
			nflop = getInstNFLOP(instructionType, opcode, output);
			return getCPUTime(nflop, metrics, output);
		}
		nflop = getInstNFLOP(instructionType, opcode, output, inputs);
		return getCPUTime(nflop, metrics, output, inputs);
	}

	public static double getParameterizedBuiltinInstTime(ParameterizedBuiltinCPInstruction inst, VarStats input, VarStats output, IOMetrics metrics) {
		CPType instructionType = inst.getCPInstructionType();
		String opcode = inst.getOpcode();
		if (opcode.equals(Opcodes.RMEMPTY.toString())) {
			String margin = inst.getParameterMap().get("margin");
			opcode += "_" + margin;
		} else if (opcode.equals(Opcodes.GROUPEDAGG.toString())) {
			CMOperator.AggregateOperationTypes opType = ((CMOperator) inst.getOperator()).getAggOpType();
			opcode += "_" + opType.name().toLowerCase();
		}
		long nflop = getInstNFLOP(instructionType, opcode, output, input);
		return getCPUTime(nflop, metrics, output, input);
	}

	public static double getMultiReturnBuiltinInstTime(MultiReturnBuiltinCPInstruction inst, VarStats input, VarStats[] outputs, IOMetrics metrics) {
		CPType instructionType = inst.getCPInstructionType();
		String opcode = inst.getOpcode();
		long nflop = getInstNFLOP(instructionType, opcode, outputs[0], input);
		double time = getCPUTime(nflop, metrics, outputs[0], input);
		for (int i = 1; i < outputs.length; i++) {
			time += IOCostUtils.getMemWriteTime(outputs[i], metrics);
		}
		return time;
	}

	// HELPERS
	public static boolean opcodeRequiresScan(String opcode) {
		return  !opcode.equals(Opcodes.NCOL.toString()) &&
				!opcode.equals(Opcodes.NROW.toString()) &&
				!opcode.equals(Opcodes.LENGTH.toString()) &&
				!opcode.equals(Opcodes.EXISTS.toString()) &&
				!opcode.equals(Opcodes.LINEAGE.toString());
	}
	public static void assignOutputMemoryStats(CPInstruction inst, VarStats output, VarStats...inputs) {
		CPType instType = inst.getCPInstructionType();
		String opcode = inst.getOpcode();

		if (inst instanceof MultiReturnBuiltinCPInstruction) {
			boolean inferred = false;
			for (VarStats current : inputs) {
				if (!inferred && current.getCells() < 0) {
					inferStats(instType, opcode, output, inputs);
					inferred = true;
				}
				if (current.getCells() < 0) {
					throw new RuntimeException("Operation of type MultiReturnBuiltin with opcode '" + opcode + "' has incomplete formula for inferring dimensions");
				}
				current.allocatedMemory = OptimizerUtils.estimateSizeExactSparsity(current.characteristics);
			}
			return;
		} else if (output.getCells() < 0) {
			inferStats(instType, opcode, output, inputs);
		}
		output.allocatedMemory = output.isScalar()? 1 : OptimizerUtils.estimateSizeExactSparsity(output.characteristics);
	}

	public static void inferStats(CPType instType, String opcode, VarStats output, VarStats...inputs) {
		switch (instType) {
			case Unary:
			case Builtin:
				copyMissingDim(output, inputs[0]);
				break;
			case AggregateUnary:
				if (opcode.startsWith("uar")) {
					copyMissingDim(output, inputs[0].getM(), 1);
				} else if (opcode.startsWith("uac")) {
					copyMissingDim(output, 1, inputs[0].getN());
				} else {
					copyMissingDim(output, 1, 1);
				}
				break;
			case MatrixIndexing:
				if (opcode.equals("rightIndex")) {
					long rowLower = (inputs[2].varName.matches("\\d+") ? Long.parseLong(inputs[2].varName) : -1);
					long rowUpper = (inputs[3].varName.matches("\\d+") ? Long.parseLong(inputs[3].varName) : -1);
					long colLower = (inputs[4].varName.matches("\\d+") ? Long.parseLong(inputs[4].varName) : -1);
					long colUpper = (inputs[5].varName.matches("\\d+") ? Long.parseLong(inputs[5].varName) : -1);

					long rowRange;
					{
						if (rowLower > 0 && rowUpper > 0) rowRange = rowUpper - rowLower + 1;
						else if (inputs[2].varName.equals(inputs[3].varName)) rowRange = 1;
						else
							rowRange = inputs[0].getM() > 0 ? inputs[0].getM() : DEFAULT_INFERRED_DIM;
					}
					long colRange;
					{
						if (colLower > 0 && colUpper > 0) colRange = colUpper - colLower + 1;
						else if (inputs[4].varName.equals(inputs[5].varName)) colRange = 1;
						else
							colRange = inputs[0].getM() > 0 ? inputs[0].getN()  : DEFAULT_INFERRED_DIM;
					}
					copyMissingDim(output, rowRange, colRange);
				} else { // leftIndex
					copyMissingDim(output, inputs[0]);
				}
				break;
			case Reorg:
				switch (opcode) {
					case "r'":
						copyMissingDim(output, inputs[0].getN(), inputs[0].getM());
						break;
					case "rev":
						copyMissingDim(output, inputs[0]);
						break;
					case "rdiag":
						if (inputs[0].getN() == 1) // diagV2M
							copyMissingDim(output, inputs[0].getM(), inputs[0].getM());
						else // diagM2V
							copyMissingDim(output, inputs[0].getM(), 1);
						break;
					case "rsort":
						boolean ixRet = Boolean.parseBoolean(inputs[1].varName);
						if (ixRet)
							copyMissingDim(output, inputs[0].getM(), 1);
						else
							copyMissingDim(output, inputs[0]);
						break;
				}
				break;
			case Binary:
				// handle case of matrix-scalar op. with the matrix being the second operand
				VarStats origin = inputs[0].isScalar()? inputs[1] : inputs[0];
				copyMissingDim(output, origin);
				break;
			case AggregateBinary:
				boolean transposeLeft = false;
				boolean transposeRight = false;
				if (inputs.length == 4) {
					transposeLeft = inputs[2] != null && Boolean.parseBoolean(inputs[2].varName);
					transposeRight = inputs[3] != null && Boolean.parseBoolean(inputs[3].varName);
				}
				if (transposeLeft && transposeRight)
					copyMissingDim(output, inputs[0].getM(), inputs[1].getM());
				else if (transposeLeft)
					copyMissingDim(output, inputs[0].getM(), inputs[1].getN());
				else if (transposeRight)
					copyMissingDim(output, inputs[0].getN(), inputs[1].getN());
				else
					copyMissingDim(output, inputs[0].getN(), inputs[1].getM());
				break;
			case ParameterizedBuiltin:
				if (opcode.equals(Opcodes.RMEMPTY.toString()) || opcode.equals(Opcodes.REPLACE.toString())) {
					copyMissingDim(output, inputs[0]);
				} else if (opcode.equals(Opcodes.UPPERTRI.toString()) || opcode.equals(Opcodes.LOWERTRI.toString())) {
					copyMissingDim(output, inputs[0].getM(), inputs[0].getM());
				}
				break;
			case Rand:
				// inferring missing output dimensions is handled exceptionally here
				if (output.getCells() < 0) {
					long nrows = (inputs[0].varName.matches("\\d+") ? Long.parseLong(inputs[0].varName) : -1);
					long ncols = (inputs[1].varName.matches("\\d+") ? Long.parseLong(inputs[1].varName) : -1);
					copyMissingDim(output, nrows, ncols);
				}
				break;
			case Ctable:
				long m = (inputs[2].varName.matches("\\d+") ? Long.parseLong(inputs[2].varName) : -1);
				long n = (inputs[3].varName.matches("\\d+") ? Long.parseLong(inputs[3].varName) : -1);
				if (inputs[1].isScalar()) {// Histogram
					if (m < 0) m = inputs[0].getM();
					if (n < 0) n = 1;
					copyMissingDim(output, m, n);
				} else { // transform (including "ctableexpand")
					if (m < 0) m = inputs[0].getM();
					if (n < 0) n = inputs[1].getCells();  // NOTE: very generous assumption, it could be revised;
					copyMissingDim(output, m, n);
				}
				break;
			case MultiReturnBuiltin:
				// special case: output and inputs stats arguments are swapped: always single input with multiple outputs
				VarStats FirstStats = inputs[0];
				VarStats SecondStats = inputs[1];
				switch (opcode) {
					case "qr":
						copyMissingDim(FirstStats, output.getM(), output.getM()); // Q
						copyMissingDim(SecondStats, output.getM(), output.getN()); // R
						break;
					case "lu":
						copyMissingDim(FirstStats, output.getN(), output.getN()); // L
						copyMissingDim(SecondStats, output.getN(), output.getN()); // U
						break;
					case "eigen":
						copyMissingDim(FirstStats, output.getN(), 1); // values
						copyMissingDim(SecondStats, output.getN(), output.getN()); // vectors
						break;
					// not all opcodes supported yet
				}
				break;
			default:
				throw new RuntimeException("Operation of type "+instType+" with opcode '"+opcode+"' has no formula for inferring dimensions");
		}
		if (output.getCells() < 0) {
			throw new RuntimeException("Operation of type "+instType+" with opcode '"+opcode+"' has incomplete formula for inferring dimensions");
		}
		if (output.getNNZ() < 0) {
			output.characteristics.setNonZeros(output.getCells());
		}
	}

	private static void copyMissingDim(VarStats target, long originRows, long originCols) {
		if (target.getM() < 0)
			target.characteristics.setRows(originRows);
		if (target.getN() < 0)
			target.characteristics.setCols(originCols);
	}

	private static void copyMissingDim(VarStats target, VarStats origin) {
		if (target.getM() < 0)
			target.characteristics.setRows(origin.getM());
		if (target.getN() < 0)
			target.characteristics.setCols(origin.getN());
	}

	public static double getCPUTime(long nflop, IOCostUtils.IOMetrics driverMetrics, VarStats output, VarStats...inputs) {
		double memScanTime = 0;
		for (VarStats input: inputs) {
			if (input == null) continue;
			memScanTime += IOCostUtils.getMemReadTime(input, driverMetrics);
		}
		double cpuComputationTime = (double) nflop / driverMetrics.cpuFLOPS;
		double memWriteTime = output != null? IOCostUtils.getMemWriteTime(output, driverMetrics) : 0;
		return Math.max(memScanTime, cpuComputationTime) + memWriteTime;
	}

	/**
	 *
	 * @param instructionType instruction type
	 * @param opcode instruction opcode, potentially with suffix to mark an extra op. characteristic
	 * @param output output's variable statistics, null is not needed for the estimation
	 * @param inputs any inputs' variable statistics, no object passed is not needed for estimation
	 * @return estimated number of floating point operations
	 */
	public static long getInstNFLOP(
			CPType instructionType,
			String opcode,
			VarStats output,
			VarStats...inputs
	) {
		opcode = opcode.toLowerCase(); // enforce lowercase for convince
		long m;
		double costs = 0;
		switch (instructionType) {
			// types corresponding to UnaryCPInstruction
			case Unary:
			case Builtin: // log and log_nz only
				if (output == null || inputs.length < 1)
					throw new RuntimeException("Not all required arguments for Unary/Builtin operations are passed initialized");
				double sparsity = inputs[0].getSparsity();
				switch (opcode) {
					case "!":
					case "isna":
					case "isnan":
					case "isinf":
					case "ceil":
					case "floor":
						costs = 1;
						break;
					case "abs":
					case "round":
					case "sign":
						costs = 1 * sparsity;
						break;
					case "sprop":
					case "sqrt":
						costs = 2 * sparsity;
						break;
					case "exp":
						costs = 18 * sparsity;
						break;
					case "sigmoid":
						costs = 21 * sparsity;
						break;
					case "log":
						costs = 32;
						break;
					case "log_nz":
					case "plogp":
						costs = 32 * sparsity;
						break;
					case "print":
					case "assert":
						costs = 1;
						break;
					case "sin":
						costs = 18 * sparsity;
						break;
					case "cos":
						costs = 22 * inputs[0].getSparsity();
						break;
					case "tan":
						costs = 42 * inputs[0].getSparsity();
						break;
					case "asin":
					case "sinh":
						costs = 93;
						break;
					case "acos":
					case "cosh":
						costs = 103;
						break;
					case "atan":
					case "tanh":
						costs = 40;
						break;
					case "ucumk+":
					case "ucummin":
					case "ucummax":
					case "ucum*":
						costs = 1 * sparsity;
						break;
					case "ucumk+*":
						costs = 2 * sparsity;
						break;
					case "stop":
						costs = 0;
						break;
					case "typeof":
						costs = 1;
						break;
					case "inverse":
						costs = (4.0 / 3.0) * output.getCellsWithSparsity() * output.getCellsWithSparsity();
						break;
					case "cholesky":
						costs = (1.0 / 3.0) * output.getCellsWithSparsity() * output.getCellsWithSparsity();
						break;
					case "det":
					case "detectschema":
					case "colnames":
						throw new RuntimeException("Specific Frame operation with opcode '" + opcode + "' is not supported yet");
					default:
						// at the point of implementation no further supported operations
						throw new DMLRuntimeException("Unary operation with opcode '" + opcode + "' is not supported by SystemDS");
				}
				return (long) (costs * output.getCells());
			case AggregateUnary:
				if (output == null || inputs.length < 1)
					throw new RuntimeException("Not all required arguments for AggregateUnary operations are passed initialized");
				switch (opcode) {
					case "nrow":
					case "ncol":
					case "length":
					case "exists":
					case "lineage":
						return DEFAULT_NFLOP_NOOP;
					case "uak+":
					case "uark+":
					case "uack+":
						costs = 4;
						break;
					case "uasqk+":
					case "uarsqk+":
					case "uacsqk+":
						costs = 5;
						break;
					case "uamean":
					case "uarmean":
					case "uacmean":
						costs = 7;
						break;
					case "uavar":
					case "uarvar":
					case "uacvar":
						costs = 14;
						break;
					case "uamax":
					case "uarmax":
					case "uarimax":
					case "uacmax":
					case "uamin":
					case "uarmin":
					case "uarimin":
					case "uacmin":
						costs = 1;
						break;
					case "ua+":
					case "uar+":
					case "uac+":
					case "ua*":
					case "uar*":
					case "uac*":
						costs = 1 * output.getSparsity();
						break;
					// count distinct operations
					case "uacd":
					case "uacdr":
					case "uacdc":
					case "unique":
					case "uniquer":
					case "uniquec":
						costs = 1 * output.getSparsity();
						break;
					case "uacdap":
					case "uacdapr":
					case "uacdapc":
						costs = 0.5 * output.getSparsity(); // do not iterate through all the cells
						break;
					// aggregation over the diagonal of a square matrix
					case "uatrace":
					case "uaktrace":
						return inputs[0].getM();
					default:
						// at the point of implementation no further supported operations
						throw new DMLRuntimeException("AggregateUnary operation with opcode '" + opcode + "' is not supported by SystemDS");
				}
				// scale
				if (opcode.startsWith("uar")) {
					costs *= inputs[0].getM();
				} else if (opcode.startsWith("uac")) {
					costs *= inputs[0].getN();
				} else {
					costs *= inputs[0].getCells();
				}
				return (long) (costs * output.getCells());
			case MMTSJ:
				if (inputs.length < 1)
					throw new RuntimeException("Not all required arguments for MMTSJ operations are passed initialized");
				// reduce by factor of 4: matrix multiplication better than average FLOP count
				// + multiply only upper triangular
				if (opcode.equals("tsmm_left")) {
					costs = inputs[0].getN() * (inputs[0].getSparsity() / 2);
				} else { // tsmm/tsmm_right
					costs = inputs[0].getM() * (inputs[0].getSparsity() / 2);
				}
				return (long) (costs * inputs[0].getCellsWithSparsity());
			case Reorg:
			case Reshape:
				if (output == null)
					throw new RuntimeException("Not all required arguments for Reorg/Reshape operations are passed initialized");
				if (opcode.equals(Opcodes.SORT.toString()))
					return (long) (output.getCellsWithSparsity() * (Math.log(output.getM()) / Math.log(2))); // merge sort columns (n*m*log2(m))
				return output.getCellsWithSparsity();
			case MatrixIndexing:
				if (output == null)
					throw new RuntimeException("Not all required arguments for Indexing operations are passed initialized");
				return output.getCellsWithSparsity();
			case MMChain:
				if (inputs.length < 1)
					throw new RuntimeException("Not all required arguments for MMChain operations are passed initialized");
				// reduction by factor 2 because matrix mult better than average flop count
				//  (mmchain essentially two matrix-vector muliplications)
				return (2 + 2) * inputs[0].getCellsWithSparsity() / 2;
			case QSort:
				if (inputs.length < 1)
					throw new RuntimeException("Not all required arguments for QSort operations are passed initialized");
				// mergesort since comparator used
				m = inputs[0].getM();
				if (opcode.equals(Opcodes.QSORT.toString()))
					costs = m + m;
				else // == "qsort_wts" (with weights)
					costs = m * inputs[0].getSparsity();
				return (long) (costs + m * (int) (Math.log(m) / Math.log(2)) + m);
			case CentralMoment:
				if (inputs.length < 1)
					throw new RuntimeException("Not all required arguments for CentralMoment operations are passed initialized");
				switch (opcode) {
					case "cm_sum":
						throw new RuntimeException("Undefined behaviour for CentralMoment operation of type sum");
					case "cm_min":
					case "cm_max":
					case "cm_count":
						costs = 2;
						break;
					case "cm_mean":
						costs = 9;
						break;
					case "cm_variance":
					case "cm_cm2":
						costs = 17;
						break;
					case "cm_cm3":
						costs = 32;
						break;
					case "cm_cm4":
						costs = 52;
						break;
					case "cm_invalid":
						// type INVALID used when unknown dimensions
						throw new RuntimeException("CentralMoment operation of type INVALID is not supported");
					default:
						// at the point of implementation no further supported operations
						throw new DMLRuntimeException("CentralMoment operation with type (<opcode>_<type>) '" + opcode + "' is not supported by SystemDS");
				}
				return (long) costs * inputs[0].getCellsWithSparsity();
			case UaggOuterChain:
			case Dnn:
				throw new RuntimeException("CP operation type'" + instructionType + "' is not supported yet");
			// types corresponding to BinaryCPInstruction
			case Binary:
				if (opcode.equals(Opcodes.PLUS.toString()) || opcode.equals(Opcodes.MINUS.toString())) {
					if (inputs.length < 2)
						throw new RuntimeException("Not all required arguments for Binary operations +/- are passed initialized");
					return inputs[0].getCellsWithSparsity() + inputs[1].getCellsWithSparsity();
				} else if (opcode.equals(Opcodes.SOLVE.toString())) {
					if (inputs.length < 1)
						throw new RuntimeException("Not all required arguments for Binary operation 'solve' are passed initialized");
					return inputs[0].getCells() * inputs[0].getN();
				}
				if (output == null)
					throw new RuntimeException("Not all required arguments for Binary operations are passed initialized");
				switch (opcode) {
					case "*":
					case "^2":
					case "*2":
					case "max":
					case "min":
					case "-nz":
					case "==":
					case "!=":
					case "<":
					case ">":
					case "<=":
					case ">=":
					case "&&":
					case "||":
					case "xor":
					case "bitwand":
					case "bitwor":
					case "bitwxor":
					case "bitwshiftl":
					case "bitwshiftr":
						costs = 1;
						break;
					case "%/%":
						costs = 6;
						break;
					case "%%":
						costs = 8;
						break;
					case "/":
						costs = 22;
						break;
					case "log":
					case "log_nz":
						costs = 32;
						break;
					case "^":
						costs = 16;
						break;
					case "1-*":
						costs = 2;
						break;
					case "dropinvalidtype":
					case "dropinvalidlength":
					case "freplicate":
					case "valueswap":
					case "applyschema":
						throw new RuntimeException("Specific Frame operation with opcode '" + opcode + "' is not supported yet");
					default:
						// at the point of implementation no further supported operations
						throw new DMLRuntimeException("Binary operation with opcode '" + opcode + "' is not supported by SystemDS");
				}
				return (long) (costs * output.getCells());
			case AggregateBinary:
				if (output == null || inputs.length < 2)
					throw new RuntimeException("Not all required arguments for AggregateBinary operations are passed initialized");
				// costs represents the cost for matrix transpose
				if (opcode.contains("_tl")) costs = inputs[0].getCellsWithSparsity();
				if (opcode.contains("_tr")) costs = inputs[1].getCellsWithSparsity();
				// else ba+*/pmm (or any of cpmm/rmm/mapmm from the Spark instructions)
				// reduce by factor of 2: matrix multiplication better than average FLOP count: 2*m*n*p->m*n*p
				return (long) (inputs[0].getN() * inputs[0].getSparsity()) * output.getCells() + (long) costs;
			case Append:
				if (inputs.length < 2)
					throw new RuntimeException("Not all required arguments for Append operation is passed initialized");
				return inputs[0].getCellsWithSparsity() + inputs[1].getCellsWithSparsity();
			case Covariance:
				if (inputs.length < 1)
					throw new RuntimeException("Not all required arguments for Covariance operation is passed initialized");
				return (long) (23 * inputs[0].getM() * inputs[0].getSparsity());
			case QPick:
				switch (opcode) {
					case "qpick_iqm":
						m = inputs[0].getM();
						return (long) (2 * m + //sum of weights
								5 * 0.25d * m + //scan to lower quantile
								8 * 0.5 * m); //scan from lower to upper quantile
					case "qpick_median":
					case "qpick_valuepick":
					case "qpick_rangepick":
						throw new RuntimeException("QuantilePickCPInstruction of operation type different from IQM is not supported yet");
					default:
						throw new DMLRuntimeException("QPick operation with opcode '" + opcode + "' is not supported by SystemDS");
				}
			// types corresponding to others CPInstruction(s)
			case Ternary:
				if (output == null)
					throw new RuntimeException("Not all required arguments for Ternary operation is passed initialized");
				switch (opcode) {
					case "+*":
					case "-*":
						return 2 * output.getCells();
					case "ifelse":
						return output.getCells();
					case "_map":
						throw new RuntimeException("Specific Frame operation with opcode '" + opcode + "' is not supported yet");
					default:
						throw new DMLRuntimeException("Ternary operation with opcode '" + opcode + "' is not supported by SystemDS");
				}
			case AggregateTernary:
				if (inputs.length < 1)
					throw new RuntimeException("Not all required arguments for AggregateTernary operation is passed initialized");
				if (opcode.equals(Opcodes.TAKPM.toString()) || opcode.equals(Opcodes.TACKPM.toString()))
					return 6 * inputs[0].getCellsWithSparsity();
				throw new DMLRuntimeException("AggregateTernary operation with opcode '" + opcode + "' is not supported by SystemDS");
			case Quaternary:
				if (inputs.length < 1)
					throw new RuntimeException("Not all required arguments for Quaternary operation is passed initialized");
				if (opcode.equals(Opcodes.WSLOSS.toString()) || opcode.equals(Opcodes.WDIVMM.toString()) || opcode.equals(Opcodes.WCEMM.toString())) {
					// 4 matrices used
					return 4 * inputs[0].getCells();
				} else if (opcode.equals(Opcodes.WSIGMOID.toString()) || opcode.equals(Opcodes.WUMM.toString())) {
					// 3 matrices used
					return 3 * inputs[0].getCells();
				}
				throw new DMLRuntimeException("Quaternary operation with opcode '" + opcode + "' is not supported by SystemDS");
			case BuiltinNary:
				if (output == null)
					throw new RuntimeException("Not all required arguments for BuiltinNary operation is passed initialized");
				switch (opcode) {
					case "cbind":
					case "rbind":
						return output.getCellsWithSparsity();
					case "nmin":
					case "nmax":
					case "n+":
						return inputs.length * output.getCellsWithSparsity();
					case "printf":
					case "list":
						return output.getN();
					case "eval":
						throw new RuntimeException("EvalNaryCPInstruction is not supported yet");
					default:
						throw new DMLRuntimeException("BuiltinNary operation with opcode '" + opcode + "' is not supported by SystemDS");
				}
			case Ctable:
				if (output == null)
					throw new RuntimeException("Not all required arguments for Ctable operation is passed initialized");
				if (opcode.startsWith(Opcodes.CTABLE.toString())) {
					// potential high inaccuracy due to unknown output column size
					// and inferring bound on number of elements what could lead to high underestimation
					return 3 * output.getCellsWithSparsity();
				}
				throw new DMLRuntimeException("Ctable operation with opcode '" + opcode + "' is not supported by SystemDS");
			case PMMJ:
				// currently this would never be reached since the pmm instruction uses AggregateBinary op. type
				if (output == null || inputs.length < 1)
					throw new RuntimeException("Not all required arguments for PMMJ operation is passed initialized");
				if (opcode.equals(Opcodes.PMM.toString())) {
					return (long) (inputs[0].getN() * inputs[0].getSparsity()) * output.getCells();
				}
				throw new DMLRuntimeException("PMMJ operation with opcode '" + opcode + "' is not supported by SystemDS");
			case ParameterizedBuiltin:
				// no argument validation here since the logic is not fully defined for this operation
				m = inputs[0].getM();
				switch (opcode) {
					case "contains":
					case "replace":
					case "tostring":
						return inputs[0].getCells();
					case "nvlist":
					case "cdf":
					case "invcdf":
					case "lowertri":
					case "uppertri":
					case "rexpand":
						return output.getCells();
					case "rmempty_rows":
						return (long) (inputs[0].getM() * Math.ceil(1.0d / inputs[0].getSparsity()) / 2)
								+ output.getCells();
					case "rmempty_cols":
						return (long) (inputs[0].getN() * Math.ceil(1.0d / inputs[0].getSparsity()) / 2)
								+ output.getCells();
					// opcode: "groupedagg"
					case "groupedagg_count":
					case "groupedagg_min":
					case "groupedagg_max":
						return 2 * m + m;
					case "groupedagg_sum":
						return 2 * m + 4 * m;
					case "groupedagg_mean":
						return 2 * m + 8 * m;
					case "groupedagg_cm2":
						return 2 * m + 16 * m;
					case "groupedagg_cm3":
						return 2 * m + 31 * m;
					case "groupedagg_cm4":
						return 2 * m + 51 * m;
					case "groupedagg_variance":
						return 2 * m + 16 * m;
					case "groupedagg_invalid":
						// type INVALID used when unknown dimensions
						throw new RuntimeException("ParameterizedBuiltin operation with opcode 'groupedagg' of type INVALID is not supported");
					case "tokenize":
					case "transformapply":
					case "transformdecode":
					case "transformcolmap":
					case "transformmeta":
					case "autodiff":
					case "paramserv":
						throw new RuntimeException("ParameterizedBuiltin operation with opcode '" + opcode + "' is not supported yet");
					default:
						throw new DMLRuntimeException("ParameterizedBuiltin operation with opcode '" + opcode + "' is not supported by SystemDS");
				}
			case MultiReturnBuiltin:
				if (inputs.length < 1)
					throw new RuntimeException("Not all required arguments for MultiReturnBuiltin operation is passed initialized");
				switch (opcode) {
					case "qr":
						costs = 2;
						break;
					case "lu":
						costs = 16;
						break;
					case "eigen":
					case "svd":
						costs = 32;
						break;
					case "fft":
					case "fft_linearized":
						throw new RuntimeException("MultiReturnBuiltin operation with opcode '" + opcode + "' is not supported yet");
					default:
						throw new DMLRuntimeException(" MultiReturnBuiltin operation with opcode '" + opcode + "' is not supported by SystemDS");
				}
				// scale up the nflop value to represent that the operations are executed by a single thread only
				// adapt later for fft/fft_linearized since they utilize all threads
				int cpuFactor = InfrastructureAnalyzer.getLocalParallelism();
				return (long) (cpuFactor * costs * inputs[0].getCells() * inputs[0].getN());
			case Prefetch:
			case EvictLineageCache:
			case Broadcast:
			case Local:
			case FCall:
			case NoOp:
				// not directly related to computation
				return 0;
			case Variable:
			case Rand:
			case StringInit:
				throw new RuntimeException(instructionType + " instructions are not handled by this method");
			case MultiReturnParameterizedBuiltin: // opcodes: transformencode
			case MultiReturnComplexMatrixBuiltin: // opcodes: ifft, ifft_linearized, stft, rcm
			case Compression: // opcode: compress
			case DeCompression: // opcode: decompress
				throw new RuntimeException("CP operation type'" + instructionType + "' is not supported yet");
			case TrigRemote:
			case Partition:
			case SpoofFused:
			case Sql:
				throw new RuntimeException("CP operation type'" + instructionType + "' is not planned for support");
			default:
				// no further supported CP types
				throw new DMLRuntimeException("CP operation type'" + instructionType + "' is not supported by SystemDS");
		}
	}
}
