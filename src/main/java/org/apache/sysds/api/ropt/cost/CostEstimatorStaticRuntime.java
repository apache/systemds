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

package org.apache.sysds.api.ropt.cost;

import com.google.errorprone.annotations.Var;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.LazyWriteBuffer;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.instructions.CPInstructionParser;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.SPInstructionParser;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.instructions.cp.CPInstruction.CPType;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.instructions.spark.SPInstruction.SPType;
import org.apache.sysds.runtime.instructions.spark.WriteSPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.TensorCharacteristics;
import scala.collection.immutable.IntMap;

import java.util.ArrayList;
import java.util.HashMap;

public class CostEstimatorStaticRuntime extends CostEstimator
{
	//time-conversion
	private static final long DEFAULT_FLOPS = 2L * 1024 * 1024 * 1024; //2GFLOPS
	//private static final long UNKNOWN_TIME = -1;
	
	//floating point operations
	private static final double DEFAULT_NFLOP_NOOP = 10; 
	private static final double DEFAULT_NFLOP_UNKNOWN = 1; 
	private static final double DEFAULT_NFLOP_CP = 1; 	
	private static final double DEFAULT_NFLOP_TEXT_IO = 350; 
	
	//IO READ throughput
	private static final double DEFAULT_MBS_FSREAD_BINARYBLOCK_DENSE = 200;
	private static final double DEFAULT_MBS_FSREAD_BINARYBLOCK_SPARSE = 100;
	private static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE = 150;
	public static final double DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE = 75;
	//IO WRITE throughput
	private static final double DEFAULT_MBS_FSWRITE_BINARYBLOCK_DENSE = 150;
	private static final double DEFAULT_MBS_FSWRITE_BINARYBLOCK_SPARSE = 75;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE = 120;
	private static final double DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE = 60;
	private static final double DEFAULT_MBS_HDFSWRITE_TEXT_DENSE = 40;
	private static final double DEFAULT_MBS_HDFSWRITE_TEXT_SPARSE = 30;

	private static long CP_FLOPS = DEFAULT_FLOPS;
	private static long SP_FLOPS = DEFAULT_FLOPS;

	public static void setCP_FLOPS(double gflops) {
		CP_FLOPS = ((long) (gflops * 1024 * 1024)) * 1024;
	}

	public static void setSP_FLOPS(double gflops) {
		SP_FLOPS = ((long) (gflops * 1024 * 1024)) * 1024;
	}

	@Override
	protected double getCPInstTimeEstimate(CPInstruction cpInstruction, HashMap<String, VarStats> stats) {
		if (cpInstruction instanceof VariableCPInstruction) {
			return getCPVariableInstTimeEstimate(cpInstruction, stats);
		} else if (cpInstruction instanceof UnaryCPInstruction) {
			return getCPUnaryInstTimeEstimate(cpInstruction, stats);
		} else if (cpInstruction instanceof BinaryCPInstruction) {
			return getCPBinaryInstTimeEstimate(cpInstruction, stats);
		} else if (cpInstruction instanceof AggregateTernaryCPInstruction) {
			VarStats input1Stats = stats.get(((AggregateTernaryCPInstruction) cpInstruction).input1.getName());
			return 6 * input1Stats.getCells();
		} else if (cpInstruction instanceof TernaryFrameScalarCPInstruction) {
			// TODO: put some real implementation:
			//  the idea is to take some worse case scenario since different mapping functionalities are possible
			// NOTE: maybe unite with AggregateTernaryCPInstruction since its similar but with factor of 6
			VarStats input1Stats = stats.get(((TernaryFrameScalarCPInstruction) cpInstruction).input1.getName());
			int dummyFactor = 4;
			return dummyFactor * input1Stats.getCells();
		} else if (cpInstruction instanceof QuaternaryCPInstruction) {
			// TODO: pattern specific and all 4 inputs requires
			VarStats input1Stats = stats.get(((QuaternaryCPInstruction) cpInstruction).input1.getName());
			return 4 * input1Stats.getCells() * input1Stats.getS();
		} else if (cpInstruction instanceof ScalarBuiltinNaryCPInstruction) {
			// TODO: maybe default cp nflops
			return 1;
		} else if (cpInstruction instanceof MatrixBuiltinNaryCPInstruction) {
			return getCPMatrixBuiltinNaryInstTimeEstimate(cpInstruction, stats);
		} else if (cpInstruction instanceof EvalNaryCPInstruction) {
			throw new RuntimeException("To be implemented later");
		} else if (cpInstruction instanceof MultiReturnBuiltinCPInstruction) {
			//note: they all have cubic complexity, the scaling factor refers to commons.math
			String opcode = cpInstruction.getOpcode();
			VarStats inputStats = stats.get(((MultiReturnBuiltinCPInstruction) cpInstruction).input1.getName());
			double xf = 2; //default e.g, qr
			switch (opcode) {
				case "eigen":
					xf = 32;
					break;
				case "lu":
					xf = 16;
					break;
				case "svd":
					xf = 32; // TODO - assuming worst case for now
					break;
			}
			return xf * inputStats.getCells() * inputStats.getN(); //for 1kx1k ~ 2GFLOP -> 1s
		} else if (cpInstruction instanceof CtableCPInstruction) {
			VarStats inputStat = stats.get(((CtableCPInstruction) cpInstruction).input1.getName());
			return inputStat.getCellsWithSparsity();
		} else if (cpInstruction instanceof PMMJCPInstruction) {
			// pure permutation cost: m*n but the operation considers sparsity
			throw new RuntimeException("To be implemented later");
		} else if (cpInstruction instanceof ParameterizedBuiltinCPInstruction || cpInstruction instanceof MultiReturnParameterizedBuiltinCPInstruction) {
			throw new DMLRuntimeException("Parametrized built-in instructions are not supported.");
		} else if (cpInstruction instanceof CompressionCPInstruction || cpInstruction instanceof DeCompressionCPInstruction) {
			throw new DMLRuntimeException("(De)Compression instructions are not supported.");
		} else if (cpInstruction instanceof SqlCPInstruction) {
			throw new DMLRuntimeException("SQL instructions are not supported.");
		} else {
			throw new RuntimeException("To be implemented later");
		}
	}

	private double getCPVariableInstTimeEstimate(CPInstruction cpInstruction, HashMap<String, VarStats> stats) {
		VariableCPInstruction varInst = (VariableCPInstruction) cpInstruction;
		String opcode = varInst.getOpcode();

		switch (opcode) {
			case "write":
				VarStats input = stats.get(varInst.getInput1().getName());
				String fmtStr = varInst.getInput3().getLiteral().getStringValue();
				FileFormat fmt = FileFormat.safeValueOf(fmtStr);
				double xwrite = fmt.isTextFormat() ? DEFAULT_NFLOP_TEXT_IO : DEFAULT_NFLOP_CP;
				return input.getCellsWithSparsity() * xwrite;
			case "case_as_matrix":
			case "case_as_frame":
				VarStats output = stats.get(varInst.getOutput().getName());
				return output.getCells();
			default:
				return DEFAULT_NFLOP_CP;
		}
	}

	private double getCPMatrixBuiltinNaryInstTimeEstimate(CPInstruction cpInstruction, HashMap<String, VarStats> stats) {
		MatrixBuiltinNaryCPInstruction inst = (MatrixBuiltinNaryCPInstruction) cpInstruction;
		String opcode = cpInstruction.getOpcode();
		CPOperand output = inst.getOutput();
		VarStats outputStats = stats.get(output.getName());

		switch (opcode) {
			case "nmin": case "nmax": case "n+": // for max, min plus num of cells for each matrix
				long numMatrices = 0;
				for (CPOperand input: inst.getInputs())
					if (input.isMatrix()) numMatrices++;
				return numMatrices * outputStats.getCells();
			case "rbind": case "cbind":
				return outputStats.getCells();
		}
		throw new DMLRuntimeException("Unknown opcode: "+opcode);
	}

	private double getCPBinaryInstTimeEstimate(CPInstruction inst, HashMap<String, VarStats> stats) {
		BinaryCPInstruction binInst = (BinaryCPInstruction) inst;
		String opcode = inst.getOpcode();
		VarStats input1Stats = stats.get(binInst.input1.getName());
		VarStats input2Stats = stats.get(binInst.input2.getName());
		VarStats outputStats = stats.get(binInst.output.getName());

		if (inst instanceof AppendCPInstruction) {
			return DEFAULT_NFLOP_CP*input1Stats.getCellsWithSparsity()*input2Stats.getCellsWithSparsity();
		} else if (inst instanceof AggregateBinaryCPInstruction) { // ba+*
			// TODO: formula correct?
			// NOTE: reduction by factor 2 because matrix mult better than average flop count (2*x/2 = x)
			if (!input1Stats.isSparse() && !input2Stats.isSparse())
				return input1Stats.getCells() * (input2Stats.getN()>1? input1Stats.getS() : 1.0) * input2Stats.getN();
			else if (input1Stats.isSparse() && !input2Stats.isSparse())
				return input1Stats.getCells() * input1Stats.getS() * input2Stats.getN();
			return  input1Stats.getCells() * input1Stats.getS() * input2Stats.getN() * input2Stats.getS();
		} else if (inst instanceof CovarianceCPInstruction) { // cov
			// NOTE: output always scalar, input 3 used as weights block if(allExists)
			//  same runtime for 2 and 3 inputs
			return 23 * input1Stats.getM(); //(11+3*k+)
		} else if (inst instanceof QuantilePickCPInstruction) {
			// TODO: implement - previous implementation is missing
			throw new RuntimeException("Not implemented yet.");
		} else {
			// TODO: Make sure no other cases of BinaryCPInstruction exist than the mentioned below
			// NOTE: the case for BinaryScalarScalarCPInstruction,
			//  					BinaryMatrixScalarCPInstruction,
			//  					BinaryMatrixMatrixCPInstruction,
			//  					BinaryFrameMatrixCPInstruction,
			//  					BinaryFrameFrameCPInstruction,
			//  					BinaryTensorTensorCPInstruction
			if( opcode.equals("+") || opcode.equals("-") //sparse safe
					&& (input1Stats.isSparse() || input2Stats.isSparse()))
				return input1Stats.getCellsWithSparsity() + input2Stats.getCellsWithSparsity();
			else if( opcode.equals("solve") ) //see also MultiReturnBuiltin
				return input1Stats.getCells() * input1Stats.getN(); //for 1kx1k ~ 1GFLOP -> 0.5s
			else
				return outputStats.getCells();
		}
	}

	private double getCPUnaryInstTimeEstimate(CPInstruction inst, HashMap<String, VarStats> stats) {
		UnaryCPInstruction unaryInst = (UnaryCPInstruction) inst;
		VarStats outputStats = stats.get(unaryInst.output.getName());
		String opcode = inst.getOpcode();
		
		// --- Operations associated with networking cost only ---
		// TODO: is somehow computational cost relevant for these operations
		if (inst instanceof PrefetchCPInstruction) {
			return 1;
		} else if (inst instanceof BroadcastCPInstruction) {
			return 1;
		} else if (inst instanceof EvictCPInstruction) {
			throw new DMLRuntimeException("Costing an instruction for GPU cache eviction is not supported.");
		}

		// --- Operations for data generation ---
		if( inst instanceof DataGenCPInstruction ) {
			if (opcode.equals(DataGen.RAND_OPCODE)) {
				DataGenCPInstruction rinst = (DataGenCPInstruction) inst;
				if( rinst.getMinValue() == 0.0 && rinst.getMaxValue() == 0.0 )
					return DEFAULT_NFLOP_CP; // empty matrix
				else if( rinst.getSparsity() == 1.0 && rinst.getMinValue() == rinst.getMaxValue() )
					return 8.0 * outputStats.getCells();
			} else if (opcode.equals(DataGen.SEQ_OPCODE)) {
				return DEFAULT_NFLOP_CP * outputStats.getCells();
			} else {
				throw new RuntimeException("To be implemented later");
			}
		}
		else if( inst instanceof StringInitCPInstruction ) {
			return DEFAULT_NFLOP_CP * outputStats.getCells();
		}
		
		// --- General unary ---
		VarStats inputStats = stats.get(unaryInst.input1.getName());

		if (outputStats == null) {
			outputStats = _scalarStats;
		}
		if (inputStats == null) {
			inputStats = _scalarStats;
		}

		if (inst instanceof MMTSJCPInstruction) {
			MMTSJType type = ((MMTSJCPInstruction) inst).getMMTSJType();
			if (type.isLeft()) {
				if (inputStats.isSparse()) {
					return inputStats.getM() * inputStats.getN() * inputStats.getS() * inputStats.getN() * inputStats.getS() / 2;
				} else {
					return inputStats.getM() * inputStats.getN() * inputStats.getS() * inputStats.getN() / 2;
				}
			} else {
				throw new RuntimeException("To be implemented later");
			}
		} else if (inst instanceof AggregateUnaryCPInstruction) {
			AggregateUnaryCPInstruction uainst = (AggregateUnaryCPInstruction) inst;
			AggregateUnaryCPInstruction.AUType autype = uainst.getAUType();
			if (autype != AggregateUnaryCPInstruction.AUType.DEFAULT) {
				// NOTE: tensors not supported for these operations
				if (inputStats.getDataType() == Types.DataType.TENSOR) {
					throw new DMLRuntimeException("Tensor does not support the opcode: " + opcode);
				}
				switch (autype) {
					case NROW:
					case NCOL:
					case LENGTH:
						return DEFAULT_NFLOP_NOOP;
					case COUNT_DISTINCT:
					case COUNT_DISTINCT_APPROX:
						// TODO: get real cost
						return inputStats.getCells();
					case UNIQUE:
						// TODO: get real cost
						return inputStats.getCells();
					case LINEAGE:
						// TODO: get real cost
						return DEFAULT_NFLOP_NOOP;
					case EXISTS:
						// TODO: get real cost
						return 1;
					default:
						// NOTE: not reachable - only for consistency
						return 0;
				}
			} else {
				switch (opcode) {
					case "cm":
						// TODO: extract attribute first and implement the logic then (CentralMomentCPInstruction)
						throw new RuntimeException("Not implemented yet.");
					case "uatrace":
					case "uaktrace":
						return 2 * inputStats.getCells();
					case "ua+":
					case "uar+":
					case "uac+":
						if (inputStats.isSparse())
							return inputStats.getCells() * inputStats.getS();
						else {
							return inputStats.getCells();
						}
					case "uak+":
					case "uark+":
					case "uack+":
						return 4 * inputStats.getCells(); // 1*k+
					case "uasqk+":
					case "uarsqk+":
					case "uacsqk+":
						return 5 * inputStats.getCells(); // +1 for multiplication to square term
					case "uamean":
					case "uarmean":
					case "uacmean":
						return 7 * inputStats.getCells(); // 1*k+
					case "uavar":
					case "uarvar":
					case "uacvar":
						return 14 * inputStats.getCells();
					case "uamax":
					case "uarmax":
					case "uacmax":
					case "uamin":
					case "uarmin":
					case "uacmin":
					case "uarimax":
					case "ua*":
						return inputStats.getCells();
					default:
						// TODO: consider if more special cases needed
						return 0;
				}
			}

		} else if(inst instanceof UnaryScalarCPInstruction) {
			// TODO: consider if 1 is always reasonable
			return 1;
		} else if(inst instanceof UnaryFrameCPInstruction) {
			switch (opcode) {
				case "typeOf":
					return 1;
				case "detectSchema":
					// TODO: think of a real static cost
					return 1;
				case "colnames":
					// TODO: is the number of the column reasonable result?
					return inputStats.getN();
			}
		}else if (inst instanceof UnaryMatrixCPInstruction){
			if (opcode.equals("print"))
				return 1;
			else if (opcode.equals("inverse")) {
				// TODO: implement
				return 0;
			} else if (opcode.equals("cholesky")) {
				// TODO: implement
				return 0;
			}
			// NOTE: What is xbu?
			double xbu = 1; //default for all ops
			if( opcode.equals("plogp") ) xbu = 2;
			else if( opcode.equals("round") ) xbu = 4;
			switch (opcode) { //opcodes: exp, abs, sin, cos, tan, sign, sqrt, plogp, print, round, sprop, sigmoid
				case "sin": case "tan": case "round": case "abs":
				case "sqrt": case "sprop": case "sigmoid": case "sign":
					return xbu * inputStats.getCellsWithSparsity();
				default:
					// TODO: does that apply to all valid unary matrix operators
					return xbu * inputStats.getCells();
			}
		} else if (inst instanceof ReorgCPInstruction || inst instanceof ReshapeCPInstruction) {
			return inputStats.getCellsWithSparsity();
		} else if (inst instanceof IndexingCPInstruction) {
			// NOTE: I doubt that this is formula for the cost is correct
			if (opcode.equals(RightIndex.OPCODE)) {
				// TODO: check correctness since I changed the initial formula to not use input 2
				return DEFAULT_NFLOP_CP * inputStats.getCellsWithSparsity();
			} else if (opcode.equals(LeftIndex.OPCODE)) {
				VarStats indexMatrixStats = stats.get(unaryInst.input2.getName());
				return DEFAULT_NFLOP_CP * inputStats.getCellsWithSparsity()
						+ 2 * DEFAULT_NFLOP_CP * indexMatrixStats.getCellsWithSparsity();
			}
		} else if (inst instanceof MMChainCPInstruction) {
			// NOTE: reduction by factor 2 because matrix mult better than average flop count
			//  (mmchain essentially two matrix-vector muliplications)
			return (2+2) * inputStats.getCellsWithSparsity() / 2;
		} else if (inst instanceof UaggOuterChainCPInstruction) {
			// TODO: implement - previous implementation is missing
			throw new RuntimeException("Not implemented yet.");
		} else if (inst instanceof QuantileSortCPInstruction) {
			// NOTE: mergesort since comparator used
			long m = inputStats.getM();
			double sortCosts = 0;
			if(unaryInst.input2 == null)
				sortCosts = DEFAULT_NFLOP_CP * m + m;
			else //w/ weights
				sortCosts = DEFAULT_NFLOP_CP * (inputStats.isSparse() ? m * inputStats.getS() : m);

			return sortCosts + m*(int)(Math.log(m)/Math.log(2)) + // mergesort
					DEFAULT_NFLOP_CP * m;
		} else if (inst instanceof DnnCPInstruction) {
			// TODO: implement the cost function for this
			throw new RuntimeException("Not implemented yet.");
		} 
		// NOTE: the upper cases should consider all possible scenarios for unary instructions
		throw new DMLRuntimeException("Attempt for costing unsupported unary instruction.");
	}
	
	/////////////////////
	// I/O Costs       //
	/////////////////////	
	
	/**
	 * Returns the estimated read time from HDFS. 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param dm rows?
	 * @param dn columns?
	 * @param ds sparsity factor?
	 * @return estimated HDFS read time
	 */
	private static double getHDFSReadTime( long dm, long dn, double ds )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		double ret = ((double)MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn))) / (1024*1024);
		
		if( sparse )
			ret /= DEFAULT_MBS_HDFSREAD_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_HDFSREAD_BINARYBLOCK_DENSE;
		
		return ret;
	}
	
	private static double getHDFSWriteTime( long dm, long dn, double ds )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double bytes = MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn));
		double mbytes = bytes / (1024*1024);
		
		double ret = -1;
		if( sparse )
			ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE;
		else //dense
			ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE;
		
		//if( LOG.isDebugEnabled() )
		//	LOG.debug("Costs[export] = "+ret+"s, "+mbytes+" MB ("+dm+","+dn+","+ds+").");
		
		
		return ret;
	}
	
	private static double getHDFSWriteTime( long dm, long dn, double ds, String format )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double bytes = MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn));
		double mbytes = bytes / (1024*1024);
		
		double ret = -1;
		
		FileFormat fmt = FileFormat.safeValueOf(format);
		if( fmt.isTextFormat() ) {
			if( sparse )
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_TEXT_SPARSE;
			else //dense
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_TEXT_DENSE;
			ret *= 2.75; //text commonly 2x-3.5x larger than binary
		}
		else {
			if( sparse )
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_SPARSE;
			else //dense
				ret = mbytes / DEFAULT_MBS_HDFSWRITE_BINARYBLOCK_DENSE;
		}
		//if( LOG.isDebugEnabled() )
		//	LOG.debug("Costs[export] = "+ret+"s, "+mbytes+" MB ("+dm+","+dn+","+ds+").");
		
		return ret;
	}

	/**
	 * Returns the estimated read time from local FS. 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param dm rows?
	 * @param dn columns?
	 * @param ds sparsity factor?
	 * @return estimated local file system read time
	 */
	public static double getFSReadTime( long dm, long dn, double ds )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double ret = ((double)MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn))) / (1024*1024);
		if( sparse )
			ret /= DEFAULT_MBS_FSREAD_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_FSREAD_BINARYBLOCK_DENSE;
		
		return ret;
	}

	public static double getFSWriteTime( long dm, long dn, double ds )
	{
		boolean sparse = MatrixBlock.evalSparseFormatOnDisk(dm, dn, (long)(ds*dm*dn));
		
		double ret = ((double)MatrixBlock.estimateSizeOnDisk(dm, dn, (long)(ds*dm*dn))) / (1024*1024);
		
		if( sparse )
			ret /= DEFAULT_MBS_FSWRITE_BINARYBLOCK_SPARSE;
		else //dense
			ret /= DEFAULT_MBS_FSWRITE_BINARYBLOCK_DENSE;
		
		return ret;
	}

	
	/////////////////////
	// Operation Costs //
	/////////////////////
	
	private static double getInstTimeEstimate(String opcode, VarStats[] vs, String[] args, ExecType et) {
		boolean inSpark = (et == ExecType.SPARK);

		return getInstTimeEstimateMatrix(opcode, inSpark,
			vs[0].getRows(), vs[0].getCols(), !vs[0]._dc.nnzKnown() ? 1.0 : vs[0].getSparsity(),
			vs[1].getRows(), vs[1].getCols(), !vs[1]._dc.nnzKnown() ? 1.0 : vs[1].getSparsity(),
			vs[2].getRows(), vs[2].getCols(), !vs[2]._dc.nnzKnown() ? 1.0 : vs[2].getSparsity(),
			args);
	}
	
	/**
	 * Returns the estimated instruction execution time, w/o data transfer and single-threaded.
	 * For scalars input dims must be set to 1 before invocation. 
	 * 
	 * NOTE: Does not handle unknowns.
	 * 
	 * @param opcode instruction opcode
	 * @param inSpark ?
	 * @param d1m ?
	 * @param d1n ?
	 * @param d1s ?
	 * @param d2m ?
	 * @param d2n ?
	 * @param d2s ?
	 * @param d3m ?
	 * @param d3n ?
	 * @param d3s ?
	 * @param args ?
	 * @return estimated instruction execution time
	 */
	private static double getInstTimeEstimate( String opcode, boolean inSpark, long d1m, long d1n, double d1s, long d2m, long d2n, double d2s, long d3m, long d3n, double d3s, String[] args)
	{
		//operation costs in seconds on single-threaded CPU
		//(excludes IO and parallelism; assumes known dims for all inputs, outputs)

		boolean leftSparse = MatrixBlock.evalSparseFormatInMemory(d1m, d1n, (long)(d1s*d1m*d1n));
		boolean rightSparse = MatrixBlock.evalSparseFormatInMemory(d2m, d2n, (long)(d2s*d2m*d2n));
		boolean onlyLeft = (d1m>=0 && d1n>=0 && d2m<0 && d2n<0 );
		boolean allExists = (d1m>=0 && d1n>=0 && d2m>=0 && d2n>=0 && d3m>=0 && d3n>=0 );

		//NOTE: all instruction types that are equivalent in CP and SP are only
		//	included in CP to prevent redundancy
		CPType cptype = CPInstructionParser.String2CPInstructionType.get(opcode);
		if( cptype != null ) //for CP Ops and equivalent MR ops
			opcode = cptype.name();
		double nflops = getNFLOP(opcode, inSpark, d1m, d1n, d1s, d2m, d2n, d2s, d3m, d3n, d3s);
		double time;
		if (inSpark) {
			time = nflops / SP_FLOPS;
		} else {
			time = nflops / CP_FLOPS;
		}

		
		if( LOG.isDebugEnabled() )
			LOG.debug("Cost["+opcode+"] = "+time+"s, "+nflops+" flops ("+d1m+","+d1n+","+d1s+","+d2m+","+d2n+","+d2s+","+d3m+","+d3n+","+d3s+").");
		
		return time;
	}
	
	private static double getNFLOP( String optype, boolean inSpark, long d1m, long d1n, double d1s, long d2m, long d2n, double d2s, long d3m, long d3n, double d3s)
	{
		//operation costs in FLOP on matrix block level (for CP and MR instructions)
		//(excludes IO and parallelism; assumes known dims for all inputs, outputs )
	
		boolean leftSparse = MatrixBlock.evalSparseFormatInMemory(d1m, d1n, (long)(d1s*d1m*d1n));
		boolean rightSparse = MatrixBlock.evalSparseFormatInMemory(d2m, d2n, (long)(d2s*d2m*d2n));
		boolean onlyLeft = (d1m>=0 && d1n>=0 && d2m<0 && d2n<0 );
		boolean allExists = (d1m>=0 && d1n>=0 && d2m>=0 && d2n>=0 && d3m>=0 && d3n>=0 );
		
		//NOTE: all instruction types that are equivalent in CP and SP are only
		//	included in CP to prevent redundancy
		CPType cptype = CPInstructionParser.String2CPInstructionType.get(optype);
		if( cptype != null ) //for CP Ops and equivalent MR ops 
		{
			//general approach: count of floating point *, /, +, -, ^, builtin ;
			switch(cptype) 
			{
			
				case AggregateBinary: //opcodes: ba+*, cov
					if( optype.equals("ba+*") ) { //matrix mult
						//reduction by factor 2 because matrix mult better than
						//average flop count
						if( !leftSparse && !rightSparse )
							return 2 * (d1m * d1n * ((d2n>1)?d1s:1.0) * d2n) /2;
						else if( !leftSparse && rightSparse )
							return 2 * (d1m * d1n * d1s * d2n * d2s) /2;
						else if( leftSparse && !rightSparse )
							return 2 * (d1m * d1n * d1s * d2n) /2;
						else //leftSparse && rightSparse
							return 2 * (d1m * d1n * d1s * d2n * d2s) /2;
					}
					else if( optype.equals("cov") ) {
						//note: output always scalar, d3 used as weights block
						//if( allExists ), same runtime for 2 and 3 inputs
						return 23 * d1m; //(11+3*k+)
					}

					return 0;
				
				case MMChain:
					//reduction by factor 2 because matrix mult better than average flop count
					//(mmchain essentially two matrix-vector muliplications)
					if( !leftSparse  )
						return (2+2) * (d1m * d1n) /2;
					else 
						return (2+2) * (d1m * d1n * d1s) /2;
					
				case AggregateTernary: //opcodes: tak+*
					return 6 * d1m * d1n; //2*1(*) + 4 (k+)
					
				case AggregateUnary: //opcodes: uak+, uark+, uack+, uasqk+, uarsqk+, uacsqk+,
				                     //         uamean, uarmean, uacmean, uavar, uarvar, uacvar,
				                     //         uamax, uarmax, uarimax, uacmax, uamin, uarmin, uacmin,
				                     //         ua+, uar+, uac+, ua*, uatrace, uaktrace,
				                     //         nrow, ncol, length, cm
					
					if( optype.equals("nrow") || optype.equals("ncol") || optype.equals("length") )
						return DEFAULT_NFLOP_NOOP;
					else if( optype.equals( "cm" ) ) {
						double xcm = 1;
						switch( Integer.parseInt(args[0]) ) {
							case 0: xcm=1; break; //count
							case 1: xcm=8; break; //mean
							case 2: xcm=16; break; //cm2
							case 3: xcm=31; break; //cm3
							case 4: xcm=51; break; //cm4
							case 5: xcm=16; break; //variance
						}
						return (leftSparse) ? xcm * (d1m * d1s + 1) : xcm * d1m;
					}
					else if( optype.equals("uatrace") || optype.equals("uaktrace") )
						return 2 * d1m * d1n;
					else if( optype.equals("ua+") || optype.equals("uar+") || optype.equals("uac+")  ){
						//sparse safe operations
						if( !leftSparse ) //dense
							return d1m * d1n;
						else //sparse
							return d1m * d1n * d1s;
					}
					else if( optype.equals("uak+") || optype.equals("uark+") || optype.equals("uack+"))
						return 4 * d1m * d1n; //1*k+
					else if( optype.equals("uasqk+") || optype.equals("uarsqk+") || optype.equals("uacsqk+"))
						return 5 * d1m * d1n; // +1 for multiplication to square term
					else if( optype.equals("uamean") || optype.equals("uarmean") || optype.equals("uacmean"))
						return 7 * d1m * d1n; //1*k+
					else if( optype.equals("uavar") || optype.equals("uarvar") || optype.equals("uacvar"))
						return 14 * d1m * d1n;
					else if(   optype.equals("uamax") || optype.equals("uarmax") || optype.equals("uacmax")
						|| optype.equals("uamin") || optype.equals("uarmin") || optype.equals("uacmin")
						|| optype.equals("uarimax") || optype.equals("ua*") )
						return d1m * d1n;
					
					return 0;
				
				case Binary: //opcodes: +, -, *, /, ^ (incl. ^2, *2),
					//max, min, solve, ==, !=, <, >, <=, >=  
					//note: all relational ops are not sparsesafe
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					if( optype.equals("+") || optype.equals("-") //sparse safe
						&& ( leftSparse || rightSparse ) )
						return d1m*d1n*d1s + d2m*d2n*d2s;
					else if( optype.equals("solve") ) //see also MultiReturnBuiltin
						return d1m * d1n * d1n; //for 1kx1k ~ 1GFLOP -> 0.5s
					else
						return d3m*d3n;
				
				case Ternary: //opcodes: +*, -*, ifelse
					return 2 * d1m * d1n;
					
				case Ctable: //opcodes: ctable
					if( optype.equals("ctable") ){
						if( leftSparse )
							return d1m * d1n * d1s; //add
						else 
							return d1m * d1n;
					}
					return 0;
				
				case Builtin: //opcodes: log 
					//note: covers scalar-scalar, scalar-matrix, matrix-matrix
					//note: can be unary or binary
					if( allExists ) //binary
						return 3 * d3m * d3n;
					else //unary
						return d3m * d3n;
					
				case Unary: //opcodes: exp, abs, sin, cos, tan, sign, sqrt, plogp, print, round, sprop, sigmoid
					//TODO add cost functions for commons math builtins: inverse, cholesky
					if( optype.equals("print") ) //scalar only
						return 1;
					else
					{
						double xbu = 1; //default for all ops
						if( optype.equals("plogp") ) xbu = 2;
						else if( optype.equals("round") ) xbu = 4;
						
						if( optype.equals("sin") || optype.equals("tan") || optype.equals("round")
							|| optype.equals("abs") || optype.equals("sqrt") || optype.equals("sprop")
							|| optype.equals("sigmoid") || optype.equals("sign") ) //sparse-safe
						{
							if( leftSparse ) //sparse
								return xbu * d1m * d1n * d1s;	
							else //dense
								return xbu * d1m * d1n;
						}
						else
							return xbu * d1m * d1n;
					}

				case Reorg: //opcodes: r', rdiag
				case Reshape: //opcodes: rshape
					if( leftSparse )
						return d1m * d1n * d1s;
					else
						return d1m * d1n;
					
				case Append: //opcodes: append
					return DEFAULT_NFLOP_CP * 
					       (((leftSparse) ? d1m * d1n * d1s : d1m * d1n ) +
					        ((rightSparse) ? d2m * d2n * d2s : d2m * d2n ));
				
				case Variable: //opcodes: assignvar, cpvar, rmvar, rmfilevar, assignvarwithfile, attachfiletovar, valuepick, iqsize, read, write, createvar, setfilename, castAsMatrix
					if( optype.equals("write") ){
						FileFormat fmt = FileFormat.safeValueOf(args[0]);
						boolean text = fmt.isTextFormat();
						double xwrite =  text ? DEFAULT_NFLOP_TEXT_IO : DEFAULT_NFLOP_CP;
						
						if( !leftSparse )
							return d1m * d1n * xwrite; 
						else
							return d1m * d1n * d1s * xwrite;
					}
					else if ( optype.equals("inmem-iqm") )
						//note: assumes uniform distribution
						return 2 * d1m + //sum of weights
						       5 + 0.25d * d1m + //scan to lower quantile
						       8 * 0.5 * d1m; //scan from lower to upper quantile
					else
						return DEFAULT_NFLOP_NOOP;
			
				case Rand: //opcodes: rand, seq
					if( optype.equals(DataGen.RAND_OPCODE) ){
						int nflopRand = 32; //per random number
						switch(Integer.parseInt(args[0])) {
							case 0: return DEFAULT_NFLOP_NOOP; //empty matrix
							case 1: return d3m * d3n * 8; //allocate, arrayfill
							case 2: //full rand
							{
								if( d3s==1.0 )
									return d3m * d3n * nflopRand + d3m * d3n * 8; //DENSE gen (incl allocate)    
								else 
									return (d3s>=MatrixBlock.SPARSITY_TURN_POINT)? 
										    2 * d3m * d3n * nflopRand + d3m * d3n * 8: //DENSE gen (incl allocate)    
									        3 * d3m * d3n * d3s * nflopRand + d3m * d3n * d3s * 24; //SPARSE gen (incl allocate)
							}
						}
					}
					else //seq
						return d3m * d3n * DEFAULT_NFLOP_CP;
				
				case StringInit: //sinit
					return d3m * d3n * DEFAULT_NFLOP_CP;
					
				case FCall: //opcodes: fcall
					//note: should be invoked independently for multiple outputs
					return d1m * d1n * d1s * DEFAULT_NFLOP_UNKNOWN;
				
				case MultiReturnBuiltin: //opcodes: qr, lu, eigen, svd
					//note: they all have cubic complexity, the scaling factor refers to commons.math
					double xf = 2; //default e.g, qr
					if( optype.equals("eigen") ) 
						xf = 32;
					else if ( optype.equals("lu") )
						xf = 16;
					else if ( optype.equals("svd"))
						xf = 32;	// TODO - assuming worst case for now
					return xf * d1m * d1n * d1n; //for 1kx1k ~ 2GFLOP -> 1s
					
				case ParameterizedBuiltin: //opcodes: cdf, invcdf, groupedagg, rmempty
					if( optype.equals("cdf") || optype.equals("invcdf"))
						return DEFAULT_NFLOP_UNKNOWN; //scalar call to commons.math
					else if( optype.equals("groupedagg") ){	
						double xga = 1;
						switch( Integer.parseInt(args[0]) ) {
							case 0: xga=4; break; //sum, see uk+
							case 1: xga=1; break; //count, see cm
							case 2: xga=8; break; //mean
							case 3: xga=16; break; //cm2
							case 4: xga=31; break; //cm3
							case 5: xga=51; break; //cm4
							case 6: xga=16; break; //variance
						}						
						return 2 * d1m + xga * d1m; //scan for min/max, groupedagg
					}	
					else if( optype.equals("rmempty") ){
						switch(Integer.parseInt(args[0])){
							case 0: //remove rows
								return ((leftSparse) ? d1m : d1m * Math.ceil(1.0d/d1s)/2) +
									   DEFAULT_NFLOP_CP * d3m * d2m;
							case 1: //remove cols
								return d1n * Math.ceil(1.0d/d1s)/2 + 
								       DEFAULT_NFLOP_CP * d3m * d2m;
						}
						
					}	
					return 0;
					
				case QSort: //opcodes: sort
					if( optype.equals("sort") ){
						//note: mergesort since comparator used
						double sortCosts = 0;
						if( onlyLeft )
							sortCosts = DEFAULT_NFLOP_CP * d1m + d1m;
						else //w/ weights
							sortCosts = DEFAULT_NFLOP_CP * ((leftSparse)?d1m*d1s:d1m); 
						return sortCosts + d1m*(int)(Math.log(d1m)/Math.log(2)) + //mergesort
										   DEFAULT_NFLOP_CP * d1m;
					}
					return 0;
					
				case MatrixIndexing: //opcodes: rightIndex, leftIndex
					if( optype.equals(LeftIndex.OPCODE) ){
						return DEFAULT_NFLOP_CP * ((leftSparse)? d1m*d1n*d1s : d1m*d1n)
						       + 2 * DEFAULT_NFLOP_CP * ((rightSparse)? d2m*d2n*d2s : d2m*d2n );
					}
					else if( optype.equals(RightIndex.OPCODE) ){
						return DEFAULT_NFLOP_CP * ((leftSparse)? d2m*d2n*d2s : d2m*d2n );
					}
					return 0;
					
				case MMTSJ: //opcodes: tsmm
					//diff to ba+* only upper triangular matrix
					//reduction by factor 2 because matrix mult better than
					//average flop count
					if( MMTSJType.valueOf(args[0]).isLeft() ) { //lefttranspose
						if( !rightSparse ) //dense
							return d1m * d1n * d1s * d1n /2;
						else //sparse
							return d1m * d1n * d1s * d1n * d1s /2; 
					}
					else if(onlyLeft) { //righttranspose
						if( !leftSparse ) //dense
							return (double)d1m * d1n * d1m /2;
						else //sparse
							return   d1m * d1n * d1s //reorg sparse
							       + d1m * d1n * d1s * d1n * d1s /2; //core tsmm
					}
					return 0;
				
				case Partition:
					return d1m * d1n * d1s + //partitioning costs
						   (inSpark ? 0 : //include write cost if in CP
							getHDFSWriteTime(d1m, d1n, d1s)* DEFAULT_FLOPS);
				
				default: 
					throw new DMLRuntimeException("CostEstimator: unsupported instruction type: "+optype);
			}
		}

		//if not found in CP instructions
		SPType sptype = SPInstructionParser.String2SPInstructionType.get(optype);
		if ( sptype != null ) //for specific Spark ops
		{
			switch (sptype) {
				case MAPMM:
				case MAPMMCHAIN:
				case TSMM2:
				case CPMM:
				case RMM:
				case ZIPMM:
				case PMAPMM:
				case MatrixIndexing:
				case Binary:
				case Reblock:
				case CSVReblock:
				case LIBSVMReblock:
				case Checkpoint:
				case ParameterizedBuiltin:
				case MAppend:
				case RAppend:
				case GAppend:
				case GAlignedAppend:
				case Quaternary:
				case CumsumAggregate:
				case CumsumOffset:
				case BinUaggChain:
				case Cast:
					return 0.0;
				default:
					throw new DMLRuntimeException("CostEstimator: unsupported instruction type: "+optype);
			}
		}
		else
		{
			throw new DMLRuntimeException("CostEstimator: unsupported instruction type: "+optype);
		}
	}
}
