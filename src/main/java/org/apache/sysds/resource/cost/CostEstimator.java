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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.DataGen;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.MMTSJ;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.*;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.instructions.spark.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

import static org.apache.sysds.common.Types.FileFormat.BINARY;
import static org.apache.sysds.common.Types.FileFormat.TEXT;
import static org.apache.sysds.lops.Data.PREAD_PREFIX;
import static org.apache.sysds.resource.cost.IOCostUtils.HDFS_SOURCE_IDENTIFIER;
import static org.apache.sysds.resource.cost.IOCostUtils.S3_SOURCE_IDENTIFIER;

import java.util.*;

/**
 * Class for estimating the execution time of a program.
 * For estimating the time for new set of resources,
 * a new instance of CostEstimator should be created.
 * TODO: consider reusing of some parts of the computation
 *   for small changes in the resources
 */
public class CostEstimator
{
	protected static final Log LOG = LogFactory.getLog(CostEstimator.class.getName());
	
	private static final int DEFAULT_NUMITER = 15;

	//time-conversion
	private static final long DEFAULT_FLOPS = 2L * 1024 * 1024 * 1024; //2GFLOPS
	//private static final long UNKNOWN_TIME = -1;

	//floating point operations
	private static final double DEFAULT_NFLOP_NOOP = 10;
	private static final double DEFAULT_NFLOP_UNKNOWN = 1;
	private static final double DEFAULT_NFLOP_CP = 1;
	private static final double DEFAULT_NFLOP_TEXT_IO = 350;

	protected static long CP_FLOPS = DEFAULT_FLOPS;
	protected static long SP_FLOPS = DEFAULT_FLOPS;
	protected static final VarStats _unknownStats = new VarStats(new MatrixCharacteristics(-1,-1,-1,-1));

	// Non-static members
	private Program _program;
	private SparkExecutionContext.MemoryManagerParRDDs _parRDDs;
	@SuppressWarnings("unused")
	private double[] cpCost; // (compute cost, I/O cost) for CP instructions
	@SuppressWarnings("unused")
	private double[] spCost; // (compute cost, I/O cost) for Spark instructions
	// declare here the hashmaps
	private final HashMap<String, VarStats> _stats;
	private HashMap<String, RDDStats> _sparkStats;
	private LinkedList<SPInstruction> _transformations;
	private final HashSet<String> _functions;
	private final long localMemory;
	private long usedMememory;

	/**
	 * Entry point for estimating the execution time of a program.
	 * @param program compiled runtime program
	 * @return estimated time for execution of the program
	 * given the resources set in {@link SparkExecutionContext}
	 * @throws CostEstimationException in case of errors
	 */
	public static double estimateExecutionTime(Program program) throws CostEstimationException {
		CostEstimator estimator = new CostEstimator(program);
		double costs = estimator.getTimeEstimate();
		return costs;
	}
	public CostEstimator(Program program) {
		_program = program;
		// initialize here the hashmaps
		_stats = new HashMap<>();
		_sparkStats = new HashMap<>();
		_transformations = new LinkedList<>();
		_functions = new HashSet();
		localMemory = (long) OptimizerUtils.getLocalMemBudget();
		this._parRDDs = new SparkExecutionContext.MemoryManagerParRDDs(0.1);
		usedMememory = 0;
		cpCost = new double[]{0.0, 0.0};
		spCost = new double[]{0.0, 0.0};
	}

	/**
	 * Meant to be used for testing purposes
	 */
	public void putStats(HashMap<String, VarStats> inputStats) {
		_stats.putAll(inputStats);
	}

	/**
	 * Intended to be called only when it is certain that the corresponding
	 * variable is not a scalar and its statistics are in {@code _stats} already.
	 * @param statsName the corresponding operand name
	 * @return {@code VarStats object} if the given key is present
	 * in the map saving the current variable statistics.
	 * @throws RuntimeException if the corresponding variable is not in {@code _stats}
	 */
	public VarStats getStats(String statsName) {
		VarStats result = _stats.get(statsName);
		if (result == null) {
			throw new RuntimeException(statsName+" key not imported yet");
		}
		return result;
	}

	/**
	 * Intended to be called when the corresponding variable could be scalar.
	 * @param statsName the corresponding operand name
	 * @return {@code VarStats object} in any case
	 */
	public VarStats getStatsWithDefaultScalar(String statsName) {
		VarStats result = _stats.get(statsName);
		if (result == null) {
			result = new VarStats(null);
		}
		return result;
	}

	public static void setCP_FLOPS(long gFlops) {
		CP_FLOPS = gFlops;
	}
	public static void setSP_FLOPS(long gFlops) {
		SP_FLOPS = gFlops;
	}

	public double getTimeEstimate() throws CostEstimationException {
		double costs = 0;

		//get cost estimate
		for( ProgramBlock pb : _program.getProgramBlocks() )
			costs += getTimeEstimatePB(pb);

		return costs;
	}

	private double getTimeEstimatePB(ProgramBlock pb) throws CostEstimationException {
		double ret = 0;

		if (pb instanceof WhileProgramBlock) {
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				ret += getTimeEstimatePB(pb2);
			ret *= DEFAULT_NUMITER;
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock tmp = (IfProgramBlock)pb; {
			for( ProgramBlock pb2 : tmp.getChildBlocksIfBody() )
				ret += getTimeEstimatePB(pb2);
			if( tmp.getChildBlocksElseBody()!=null )
				for( ProgramBlock pb2 : tmp.getChildBlocksElseBody() ){
					ret += getTimeEstimatePB(pb2);
					ret /= 2; //weighted sum
				}
			}
		}
		else if (pb instanceof ForProgramBlock) { //includes ParFORProgramBlock
			ForProgramBlock tmp = (ForProgramBlock)pb;
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				ret += getTimeEstimatePB(pb2);
			// NOTE: currently ParFor blocks are handled as regular for block
			//  what could lead to very inaccurate estimation in case of complex ParFor blocks
			ret *= OptimizerUtils.getNumIterations(tmp, DEFAULT_NUMITER);
		}
		else if ( pb instanceof FunctionProgramBlock ) {
			FunctionProgramBlock tmp = (FunctionProgramBlock) pb;
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				ret += getTimeEstimatePB(pb2);
		}
		else if( pb instanceof BasicProgramBlock )
		{
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			ArrayList<Instruction> tmp = bpb.getInstructions();

			for( Instruction inst : tmp )
			{
				if( inst instanceof FunctionCallCPInstruction ) //functions
				{
					FunctionCallCPInstruction finst = (FunctionCallCPInstruction)inst;
					String fkey = DMLProgram.constructFunctionKey(finst.getNamespace(), finst.getFunctionName());
					//awareness of recursive functions, missing program
					if( !_functions.contains(fkey) && pb.getProgram()!=null )
					{
						_functions.add(fkey);
						maintainFCallInputStats(finst);
						FunctionProgramBlock fpb = _program.getFunctionProgramBlock(fkey, true);
						ret = getTimeEstimatePB(fpb);
						maintainFCallOutputStats(finst, fpb);
						_functions.remove(fkey);
					}
				} else {
					maintainStats(inst);
					ret += getTimeEstimateInst(inst);
				}
			}
		}
		return ret;
	}

	/**
	 * Creates copies of the {@code VarStats} for the function argument.
	 * Meant to be called before estimating the execution time of
	 * the function program block of the corresponding function call instruction,
	 * otherwise the relevant statistics would not be available for the estimation.
	 */
	public void maintainFCallInputStats(FunctionCallCPInstruction finst) {
		CPOperand[] inputs = finst.getInputs();
		for (int i = 0; i < inputs.length; i++) {
			DataType dt = inputs[i].getDataType();
			if (dt == DataType.TENSOR) {
				throw new DMLRuntimeException("Tensor is not supported for cost estimation");
			} else if (dt == DataType.MATRIX || dt == DataType.FRAME || dt == DataType.LIST) {
				VarStats argStats = getStats(inputs[i].getName());
				argStats.refCount++;
				_stats.put(finst.getFunArgNames().get(i), argStats);
			}
			// ignore scalars
		}
	}

	/**
	 * Creates copies of the {@code VarStats} for the function output parameters.
	 * Meant to be called after estimating the execution time of
	 * the function program block of the corresponding function call instruction,
	 * otherwise the relevant statistics would not have been created yet.
	 */
	public void maintainFCallOutputStats(FunctionCallCPInstruction finst, FunctionProgramBlock fpb) {
		List<DataIdentifier> params = fpb.getOutputParams();
		List<String> boundNames = finst.getBoundOutputParamNames();
		for(int i = 0; i < params.size(); i++) {
			DataType dt = params.get(i).getDataType();
			if (dt == DataType.TENSOR) {
				throw new DMLRuntimeException("Tensor is not supported for cost estimation");
			}
			else if (dt == DataType.MATRIX || dt == DataType.FRAME || dt == DataType.LIST) {
				VarStats boundStats = getStats(params.get(i).getName());
				boundStats.refCount++;
				_stats.put(boundNames.get(i), boundStats);
			}
			// ignore scalars
		}
	}

	/**
	 * Keep the basic-block variable statistics updated and compute I/O cost.
	 * NOTE: At program execution reading the files is done once
	 * 	the matrix is needed but cost estimation the place for
	 * 	adding cost is not relevant.
	 */
	public void maintainStats(Instruction inst) throws CostEstimationException {
		// CP Instructions changing the map for statistics
		if(inst instanceof VariableCPInstruction)
		{
			String opcode = inst.getOpcode();
			VariableCPInstruction vinst = (VariableCPInstruction) inst;
			if (vinst.getInput1().getDataType() == DataType.TENSOR) {
				throw new DMLRuntimeException("Tensor is not supported for cost estimation");
			}
			String varName = vinst.getInput1().getName();
			switch (opcode) {
				case "createvar":
					DataCharacteristics dataCharacteristics = vinst.getMetaData().getDataCharacteristics();
					VarStats varStats = new VarStats(dataCharacteristics);
					varStats.isDirty = true;
					if (vinst.getInput1().getName().startsWith(PREAD_PREFIX)) {
						// NOTE: add I/O here although at execution the reading is done when the input is needed
						String fileName = vinst.getInput2().getName();
						String dataSource = IOCostUtils.getDataSource(fileName);
						varStats.fileInfo = new Object[]{dataSource, ((MetaDataFormat) vinst.getMetaData()).getFileFormat()};
					}
					_stats.put(varName, varStats);
					break;
				case "cpvar":
					VarStats outputStats = _stats.get(vinst.getInput2().getName());
					// handle the case of the output variable was loaded in memory; does nothing if null
					removeFromMemory(outputStats);
					outputStats = getStats(varName);
					_stats.put(vinst.getInput2().getName(), outputStats);
					outputStats.refCount++;
					break;
				case "mvvar":
					VarStats statsToMove = _stats.remove(varName);
					_stats.put(vinst.getInput2().getName(), statsToMove);
					break;
				case "rmvar":
					for (CPOperand inputOperand: vinst.getInputs()) {
						VarStats inputVar = _stats.remove(inputOperand.getName());
						if (inputVar != null && --inputVar.refCount < 1) { // inputVar == null for scalars
							removeFromMemory(inputVar);
						}
					}
					break;
			}
		}
		else if (inst instanceof DataGenCPInstruction){
			// variable already created at "createvar"
			// now update the sparsity and set size estimate
			String opcode = inst.getOpcode();
			if (opcode.equals("rand")) {
				DataGenCPInstruction dinst = (DataGenCPInstruction) inst;
				VarStats stat = getStats(dinst.getOutput().getName());
				stat.characteristics.setNonZeros((long) (stat.getCells()*dinst.getSparsity()));
			}
		} else if (inst instanceof ReblockSPInstruction || inst instanceof CSVReblockSPInstruction) {
			VarStats input = getStats(((UnarySPInstruction) inst).input1.getName());
			VarStats output = getStats(((UnarySPInstruction) inst).getOutputVariableName());
			if (input.allocatedMemory < 0 && input.fileInfo == null) {
				throw new RuntimeException("Reblock instruction allowed only loaded variables or for such with given source file");
			}
			// re-assigning to null is fine as long as the allocated memory is non-negative
			output.fileInfo = input.fileInfo;
//			output.rddStats = new RDDStats(output); NOTE: no stats so it can be detected if used or not when loading

		} else if (inst instanceof CheckpointSPInstruction) {
			// TODO: consider how to handle chkpoint since also potentially not triggered: maybe a flag in cpStats tho be read at RDDStats init.
			// NOTE: maybe also file info copy: decide how ti handle binary read in Spark
			VarStats input = getStats(((CheckpointSPInstruction) inst).input1.getName());
			_stats.put(((CheckpointSPInstruction) inst).output.getName(), input);
			//output.setCheckpoint = true;
		} else if (inst instanceof SPInstruction) {
			// else if statement on purpose after reblock and checkpoint instructions
			if (inst instanceof WriteSPInstruction) {
				// WriteSPInstruction the only SPInstruction without output
				return;
			}
			VarStats output;
			if (inst instanceof ComputationSPInstruction) {
				output = getStats(((ComputationSPInstruction) inst).getOutputVariableName());
			} else if (inst instanceof BuiltinNarySPInstruction) {
				output = getStats(((BuiltinNarySPInstruction) inst).output.getName());
			} else if (inst instanceof MapmmChainSPInstruction) {
				output = getStats(((MapmmChainSPInstruction) inst).output.getName());
			} else if (inst instanceof SpoofSPInstruction) {
				throw new RuntimeException("SpoofSPInstruction is not supported for cost estimation.");
			} else {
				// not other SPInstruction implementations currently but
				// add exception avoid unclear state in case of new instructions implemented in the future
				throw new RuntimeException(((SPInstruction) inst).getSPInstructionType() + " instruction not supported yet.");
			}
			output.rddStats = new RDDStats(output);
		}
	}

	public double getTimeEstimateInst(Instruction inst) throws CostEstimationException {
		double ret = 0;
		if (inst instanceof CPInstruction) {
			ret = getTimeEstimateCPInst((CPInstruction)inst);
		} else { // inst instanceof SPInstruction
			_transformations.add((SPInstruction) inst);
		}
		return ret;
	}

	/**
	 * Estimates the execution time of a single CP instruction
	 * following the formula <i>C(p) = T_w + max(T_r, T_c)</i> with:
	 * <li>T_w - instruction write (to mem.) time</li>
	 * <li>T_r - instruction read (to mem.) time</li>
	 * <li>T_c - instruction compute time</li>
	 *
	 * @param inst
	 * @return
	 * @throws CostEstimationException
	 */
	public double getTimeEstimateCPInst(CPInstruction inst) throws CostEstimationException {
		double ret = 0;
		if (inst instanceof VariableCPInstruction) {
			String opcode = inst.getOpcode();
			VariableCPInstruction varInst = (VariableCPInstruction) inst;
			VarStats input = getStatsWithDefaultScalar(varInst.getInput1().getName());
			if (opcode.startsWith("cast")) {
				ret += getLoadTime(input); // disk I/O estimate
				double scanTime = IOCostUtils.getMemReadTime(input); // memory read cost
				double computeTime = getNFLOP_CPVariableInst(varInst, input) / CP_FLOPS;
				ret += Math.max(scanTime, computeTime);
				CPOperand outputOperand = varInst.getOutput();
				VarStats output = getStatsWithDefaultScalar(outputOperand.getName());
				putInMemory(output);
				ret += IOCostUtils.getMemWriteTime(input); // memory write cost
			}
			else if (opcode.equals("write")) {
				ret += getLoadTime(input); // disk I/O estimate
				// NOTE: if the given output filename is not liter, the assumed data source would be HDFS
				String fileName = varInst.getInput2().isLiteral()? varInst.getInput2().getLiteral().getStringValue() : "hdfs_file";
				String dataSource = IOCostUtils.getDataSource(fileName);
				String formatString = varInst.getInput3().getLiteral().getStringValue();
				ret += getNFLOP_CPVariableInst(varInst, input) / CP_FLOPS; // compute time cost
				ret += IOCostUtils.getWriteTime(input.getM(), input.getN(), input.getS(),
						dataSource, FileFormat.safeValueOf(formatString)); // I/O estimate
			}

			return ret;
		}
		else if (inst instanceof UnaryCPInstruction) {
			// --- Operations associated with networking cost only ---
			// TODO: is somehow computational cost relevant for these operations
			if (inst instanceof PrefetchCPInstruction) {
				throw new DMLRuntimeException("TODO");
			} else if (inst instanceof BroadcastCPInstruction) {
				throw new DMLRuntimeException("TODO");
			} else if (inst instanceof EvictCPInstruction) {
				throw new DMLRuntimeException("Costing an instruction for GPU cache eviction is not supported.");
			}

			// --- Operations that does not require estimation ---
			if (inst.getOpcode().equals("print")) {
				return 0;
			}

			UnaryCPInstruction unaryInst = (UnaryCPInstruction) inst;
			VarStats output = getStatsWithDefaultScalar(unaryInst.getOutput().getName());
			// unary instructions that doest not require a matrix/frame input
			if (inst instanceof DataGenCPInstruction || inst instanceof StringInitCPInstruction) {
				double computeTime = getNFLOP_DataGenCPInst(unaryInst, output) / CP_FLOPS;
				ret += computeTime;
				putInMemory(output);
				ret += IOCostUtils.getMemWriteTime(output);
				return ret;
			}
			// unary instructions taking one matrix/frame as input (the rest of type UnaryCPInstruction)
			VarStats input = getStatsWithDefaultScalar(unaryInst.input1.getName());
			ret += getLoadTime(input);
			double scanTime = IOCostUtils.getMemReadTime(input);
			double computeTime = getNFLOP_CPUnaryInst(unaryInst, input) / CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			putInMemory(output);
			ret += IOCostUtils.getMemWriteTime(output);
			return ret;
		}
		else if (inst instanceof BinaryCPInstruction) {
			BinaryCPInstruction binInst = (BinaryCPInstruction) inst;
			VarStats input1 = getStatsWithDefaultScalar(binInst.input1.getName());
			VarStats input2 = getStatsWithDefaultScalar(binInst.input2.getName());
			VarStats output = getStatsWithDefaultScalar(binInst.output.getName());

			ret += getLoadTime(input1);
			ret += getLoadTime(input2);
			double scanTime = IOCostUtils.getMemReadTime(input1) + IOCostUtils.getMemReadTime(input2);
			double computeTime = getNFLOP_CPBinaryInst(binInst, input1, input2, output) / CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			putInMemory(output);
			ret += IOCostUtils.getMemWriteTime(output);
			return ret;
		}
		else if (inst instanceof AggregateTernaryCPInstruction) {
			AggregateTernaryCPInstruction aggInst = (AggregateTernaryCPInstruction) inst;
			VarStats input = getStats(aggInst.input1.getName());
			VarStats output = getStats(aggInst.getOutput().getName());

			ret += getLoadTime(input);
			double scanTime = IOCostUtils.getMemReadTime(input);
			double computeTime = (double) (6 * input.getCells()) / CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			putInMemory(output);
			ret += IOCostUtils.getMemWriteTime(output);
			return ret;
		}
		else if (inst instanceof TernaryCPInstruction) {
			// TODO: put some real implementation:
			//  the idea is to take some worse case scenario since different mapping functionalities are possible
			// NOTE: maybe unite with AggregateTernaryCPInstruction since its similar but with factor of 6
			TernaryCPInstruction tinst = (TernaryCPInstruction) inst;
			VarStats input1 = getStatsWithDefaultScalar(tinst.input1.getName());
			VarStats input2 = getStatsWithDefaultScalar(tinst.input2.getName());
			VarStats input3 = getStatsWithDefaultScalar(tinst.input3.getName());
			VarStats output = getStatsWithDefaultScalar(tinst.output.getName());

			ret += getLoadTime(input1);
			double scanTime = IOCostUtils.getMemReadTime(input1);
			double computeTime;
			if (tinst instanceof TernaryFrameScalarCPInstruction) {
				computeTime = (double) (4*input1.getCells()) / CP_FLOPS; // 4 - dummy factor
				ret += Math.max(scanTime, computeTime);
				putInMemory(output);
				ret += IOCostUtils.getMemWriteTime(output);
				return ret;
			} else { // regular TernaryCPInstruction
				ret += getLoadTime(input2) + getLoadTime(input3);
				scanTime += IOCostUtils.getMemReadTime(input2) + IOCostUtils.getMemReadTime(input3);
				computeTime = Double.MAX_VALUE; // TODO: implement
			}
			ret += Math.max(scanTime, computeTime);
			putInMemory(output);
			ret += IOCostUtils.getMemWriteTime(output);
			return ret;
		}
		else if (inst instanceof QuaternaryCPInstruction) {
			// TODO: put logical compute estimate (maybe putting a complexity factor)
			QuaternaryCPInstruction gInst = (QuaternaryCPInstruction) inst;
			VarStats input1 = getStats(gInst.input1.getName());
			VarStats input2 = getStats(gInst.input2.getName());
			VarStats input3 = getStats(gInst.input3.getName());
			VarStats input4 = getStatsWithDefaultScalar(gInst.getInput4().getName());
			VarStats output = getStats(gInst.getOutput().getName());

			ret += getLoadTime(input1) + getLoadTime(input2) + getLoadTime(input3) + getLoadTime(input4);
			double scanTime = IOCostUtils.getMemReadTime(input1)
					+ IOCostUtils.getMemReadTime(input2)
					+ IOCostUtils.getMemReadTime(input3)
					+ IOCostUtils.getMemReadTime(input4);
			double computeTime = (double) (input1.getCells() * input2.getCells() + input3.getCells() + input4.getCells())
					/ CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			putInMemory(output);
			ret += IOCostUtils.getMemWriteTime(output);
			return ret;
		}
		else if (inst instanceof ScalarBuiltinNaryCPInstruction) {
			return 1d / CP_FLOPS;
		}
		else if (inst instanceof MatrixBuiltinNaryCPInstruction) {
			MatrixBuiltinNaryCPInstruction mInst = (MatrixBuiltinNaryCPInstruction) inst;
			VarStats output = getStatsWithDefaultScalar(mInst.getOutput().getName());
			int numMatrices = 0;
			double scanTime = 0d;
			for (CPOperand operand : mInst.getInputs()) {
				if (operand.isMatrix()) {
					VarStats input = getStats(operand.getName());
					ret += getLoadTime(input);
					scanTime += IOCostUtils.getMemReadTime(input);
					numMatrices += 1;
				}

			}
			double computeTime = getNFLOP_CPMatrixBuiltinNaryInst(mInst, numMatrices, output) / CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			putInMemory(output);
			ret += IOCostUtils.getMemWriteTime(output);
			return ret;
		}
		else if (inst instanceof EvalNaryCPInstruction) {
			throw new RuntimeException("To be implemented later");
		}
		else if (inst instanceof MultiReturnBuiltinCPInstruction) {
			MultiReturnBuiltinCPInstruction mrbInst = (MultiReturnBuiltinCPInstruction) inst;
			VarStats input = getStatsWithDefaultScalar(mrbInst.input1.getName());

			ret += getLoadTime(input);
			double scanTime = IOCostUtils.getMemReadTime(input);
			double computeTime = getNFLOP_CPMultiReturnBuiltinInst(mrbInst, input) / CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			for (CPOperand operand : mrbInst.getOutputs()) {
				VarStats output = getStatsWithDefaultScalar(operand.getName());
				putInMemory(output);
				ret += IOCostUtils.getMemWriteTime(output);
			}
			return ret;
		}
		else if (inst instanceof CtableCPInstruction) {
			CtableCPInstruction ctInst = (CtableCPInstruction) inst;
			VarStats input1 = getStatsWithDefaultScalar(ctInst.input1.getName());
			VarStats input2 = getStatsWithDefaultScalar(ctInst.input2.getName());
			VarStats input3 = getStatsWithDefaultScalar(ctInst.input3.getName());

			ret += getLoadTime(input1) + getLoadTime(input2) + getLoadTime(input3);
			double scanTime = IOCostUtils.getMemReadTime(input1)
					+ IOCostUtils.getMemReadTime(input2)
					+ IOCostUtils.getMemReadTime(input3);
			double computeTime = (double) input1.getCellsWithSparsity() / CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			// TODO: figure out what dimensions to assign to the output matrix stats 'output'
			throw new DMLRuntimeException("Operation "+inst.getOpcode()+" is not supported yet due to a unpredictable output");
		}
		else if (inst instanceof PMMJCPInstruction) {
			PMMJCPInstruction pmmInst = (PMMJCPInstruction) inst;
			VarStats input1 = getStats(pmmInst.input1.getName());
			VarStats input2 = getStats(pmmInst.input2.getName());
			VarStats output = getStats(pmmInst.getOutput().getName());

			ret += getLoadTime(input1) + getLoadTime(input2);
			double scanTime = IOCostUtils.getMemReadTime(input1) + IOCostUtils.getMemReadTime(input2);
			double computeTime = input1.getCells() * input2.getCellsWithSparsity() / CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			putInMemory(output);
			ret += IOCostUtils.getMemWriteTime(output);
			return ret;
		}
		else if (inst instanceof ParameterizedBuiltinCPInstruction) {
			ParameterizedBuiltinCPInstruction paramInst = (ParameterizedBuiltinCPInstruction) inst;
			String[] parts = InstructionUtils.getInstructionParts(inst.toString());
			VarStats input = getStatsWithDefaultScalar( parts[1].substring(7).replaceAll(Lop.VARIABLE_NAME_PLACEHOLDER, "") );
			VarStats output = getStatsWithDefaultScalar( parts[parts.length-1] );

			ret += getLoadTime(input);
			double scanTime = IOCostUtils.getMemReadTime(input);
			double computeTime = getNFLOP_CPParameterizedBuiltinInst(paramInst, input, output) / CP_FLOPS;
			ret += Math.max(scanTime, computeTime);
			putInMemory(output);
			ret += IOCostUtils.getMemWriteTime(output);
			return ret;
		}
		else if (inst instanceof MultiReturnParameterizedBuiltinCPInstruction) {
			throw new DMLRuntimeException("MultiReturnParametrized built-in instructions are not supported.");
		}
		else if (inst instanceof CompressionCPInstruction || inst instanceof DeCompressionCPInstruction) {
			throw new DMLRuntimeException("(De)Compression instructions are not supported yet.");
		}
		else if (inst instanceof SqlCPInstruction) {
			throw new DMLRuntimeException("SQL instructions are not supported.");
		}
		throw new DMLRuntimeException("Unsupported instruction: " + inst.getOpcode());
	}
	private double getNFLOP_CPVariableInst(VariableCPInstruction inst, VarStats input) throws CostEstimationException {
		switch (inst.getOpcode()) {
			case "write":
				String fmtStr = inst.getInput3().getLiteral().getStringValue();
				FileFormat fmt = FileFormat.safeValueOf(fmtStr);
				double xwrite = fmt.isTextFormat() ? DEFAULT_NFLOP_TEXT_IO : DEFAULT_NFLOP_CP;
				return input.getCellsWithSparsity() * xwrite;
			case "cast_as_matrix":
			case "cast_as_frame":
				return input.getCells();
			default:
				return DEFAULT_NFLOP_CP;
		}
	}

	private double getNFLOP_DataGenCPInst(UnaryCPInstruction inst, VarStats output) {
		String opcode = inst.getOpcode();
		if( inst instanceof DataGenCPInstruction ) {
			if (opcode.equals(DataGen.RAND_OPCODE)) {
				DataGenCPInstruction rinst = (DataGenCPInstruction) inst;
				if( rinst.getMinValue() == 0.0 && rinst.getMaxValue() == 0.0 )
					return DEFAULT_NFLOP_CP; // empty matrix
				else if( rinst.getSparsity() == 1.0 && rinst.getMinValue() == rinst.getMaxValue() ) // allocate, array fill
					return 8.0 * output.getCells();
				else { // full rand
					if (rinst.getSparsity() == 1.0)
						return 32.0 * output.getCells() + 8.0 * output.getCells();//DENSE gen (incl allocate)
					if (rinst.getSparsity() < MatrixBlock.SPARSITY_TURN_POINT)
						return 3.0 * output.getCellsWithSparsity() + 24.0 * output.getCellsWithSparsity();  //SPARSE gen (incl allocate)
					return 2.0 * output.getCells() + 8.0 * output.getCells();  //DENSE gen (incl allocate)
				}
			} else if (opcode.equals(DataGen.SEQ_OPCODE)) {
				return DEFAULT_NFLOP_CP * output.getCells();
			} else {
				throw new RuntimeException("To be implemented later");
			}
		}
		else if( inst instanceof StringInitCPInstruction ) {
			return DEFAULT_NFLOP_CP * output.getCells();
		} else {
			throw new IllegalArgumentException("Method has been called with invalid instruction");
		}
	}

	private double getNFLOP_CPUnaryInst(UnaryCPInstruction inst, VarStats input) {
		String opcode = inst.getOpcode();

//		if (input == null)
//			input = _scalarStats; // TODO: consider if needed: if yes -> stats garbage collections?

		if (inst instanceof MMTSJCPInstruction) {
			MMTSJ.MMTSJType type = ((MMTSJCPInstruction) inst).getMMTSJType();
			if (type.isLeft()) {
				if (input.isSparse()) {
					return input.getM() * input.getN() * input.getS() * input.getN() * input.getS() / 2;
				} else {
					return input.getM() * input.getN() * input.getS() * input.getN() / 2;
				}
			} else {
				throw new RuntimeException("To be implemented later");
			}
		} else if (inst instanceof AggregateUnaryCPInstruction) {
			AggregateUnaryCPInstruction uainst = (AggregateUnaryCPInstruction) inst;
			AggregateUnaryCPInstruction.AUType autype = uainst.getAUType();
			if (autype != AggregateUnaryCPInstruction.AUType.DEFAULT) {
				switch (autype) {
					case NROW:
					case NCOL:
					case LENGTH:
						return DEFAULT_NFLOP_NOOP;
					case COUNT_DISTINCT:
					case COUNT_DISTINCT_APPROX:
						// TODO: get real cost
						return input.getCells();
					case UNIQUE:
						// TODO: get real cost
						return input.getCells();
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
				int k = getComputationFactorUAOp(opcode);
				if (opcode.equals("cm")) {
					// TODO: extract attribute first and implement the logic then (CentralMomentCPInstruction)
					throw new RuntimeException("Not implemented yet.");
				} else if (opcode.equals("ua+") || opcode.equals("uar+") || opcode.equals("uac+")) {
					return k*input.getCellsWithSparsity();
				} else { // NOTE: assumes all other cases were already handled properly
					return k*input.getCells();
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
					return input.getN();
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
					return xbu * input.getCellsWithSparsity();
				default:
					// TODO: does that apply to all valid unary matrix operators
					return xbu * input.getCells();
			}
		} else if (inst instanceof ReorgCPInstruction || inst instanceof ReshapeCPInstruction) {
			return input.getCellsWithSparsity(); // TODO: add FLOP dependency
		} else if (inst instanceof IndexingCPInstruction) {
			// NOTE: I doubt that this is formula for the cost is correct
			if (opcode.equals(RightIndex.OPCODE)) {
				// TODO: check correctness since I changed the initial formula to not use input 2
				return DEFAULT_NFLOP_CP * input.getCellsWithSparsity();
			} else if (opcode.equals(LeftIndex.OPCODE)) {
				VarStats indexMatrixStats = getStatsWithDefaultScalar(inst.input2.getName());
				return DEFAULT_NFLOP_CP * input.getCellsWithSparsity()
						+ 2 * DEFAULT_NFLOP_CP * indexMatrixStats.getCellsWithSparsity();
			}
		} else if (inst instanceof MMChainCPInstruction) {
			// NOTE: reduction by factor 2 because matrix mult better than average flop count
			//  (mmchain essentially two matrix-vector muliplications)
			return (2+2) * input.getCellsWithSparsity() / 2;
		} else if (inst instanceof UaggOuterChainCPInstruction) {
			// TODO: implement - previous implementation is missing
			throw new RuntimeException("Not implemented yet.");
		} else if (inst instanceof QuantileSortCPInstruction) {
			// NOTE: mergesort since comparator used
			long m = input.getM();
			double sortCosts = 0;
			if(inst.input2 == null)
				sortCosts = DEFAULT_NFLOP_CP * m + m;
			else //w/ weights
				sortCosts = DEFAULT_NFLOP_CP * (input.isSparse() ? m * input.getS() : m);

			return sortCosts + m*(int)(Math.log(m)/Math.log(2)) + // mergesort
					DEFAULT_NFLOP_CP * m;
		} else if (inst instanceof DnnCPInstruction) {
			// TODO: implement the cost function for this
			throw new RuntimeException("Not implemented yet.");
		}
		// NOTE: the upper cases should consider all possible scenarios for unary instructions
		throw new DMLRuntimeException("Attempt for costing unsupported unary instruction.");
	}

	private double getNFLOP_CPBinaryInst(BinaryCPInstruction inst, VarStats input1, VarStats input2, VarStats output) throws CostEstimationException {
		if (inst instanceof AppendCPInstruction) {
			return DEFAULT_NFLOP_CP*input1.getCellsWithSparsity()*input2.getCellsWithSparsity();
		} else if (inst instanceof AggregateBinaryCPInstruction) { // ba+*
			// TODO: formula correct?
			// NOTE: reduction by factor 2 because matrix mult better than average flop count (2*x/2 = x)
			if (!input1.isSparse() && !input2.isSparse())
				return input1.getCells() * (input2.getN()>1? input1.getS() : 1.0) * input2.getN();
			else if (input1.isSparse() && !input2.isSparse())
				return input1.getCells() * input1.getS() * input2.getN();
			return  input1.getCells() * input1.getS() * input2.getN() * input2.getS();
		} else if (inst instanceof CovarianceCPInstruction) { // cov
			// NOTE: output always scalar, input 3 used as weights block if(allExists)
			//  same runtime for 2 and 3 inputs
			return 23 * input1.getM(); //(11+3*k+)
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
			String opcode = inst.getOpcode();
			if( opcode.equals("+") || opcode.equals("-") //sparse safe
					&& (input1.isSparse() || input2.isSparse()))
				return input1.getCellsWithSparsity() + input2.getCellsWithSparsity();
			else if( opcode.equals("solve") ) //see also MultiReturnBuiltin
				return input1.getCells() * input1.getN(); //for 1kx1k ~ 1GFLOP -> 0.5s
			else
				return output.getCells();
		}
	}

	private double getNFLOP_CPMatrixBuiltinNaryInst(MatrixBuiltinNaryCPInstruction inst, int numMatrices, VarStats output) throws CostEstimationException {
		String opcode = inst.getOpcode();
		switch (opcode) {
			case "nmin": case "nmax": case "n+": // for max, min plus num of cells for each matrix
				return numMatrices * output.getCells();
			case "rbind": case "cbind":
				return output.getCells();
			default:
				throw new DMLRuntimeException("Unknown opcode: "+opcode);
		}
	}

	private double getNFLOP_CPMultiReturnBuiltinInst(MultiReturnBuiltinCPInstruction inst, VarStats input) throws CostEstimationException {
		String opcode = inst.getOpcode();
		// NOTE: they all have cubic complexity, the scaling factor refers to commons.math
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
		return xf * input.getCells() * input.getN(); //for 1kx1k ~ 2GFLOP -> 1s
	}

	private double getNFLOP_CPParameterizedBuiltinInst(ParameterizedBuiltinCPInstruction inst, VarStats input, VarStats output) throws CostEstimationException {
		String opcode = inst.getOpcode();
		switch (opcode) {
			case "toString":
				return 1;
			case "cdf":
			case "invcdf":
				return DEFAULT_NFLOP_UNKNOWN; //scalar call to commons.math
			case "groupedagg": {
				HashMap<String, String> paramsMap = inst.getParameterMap();
				String fn = paramsMap.get("fn");
				String order = paramsMap.get("order");
				CMOperator.AggregateOperationTypes type = CMOperator.getAggOpType(fn, order);
				int attr = type.ordinal();
				double xga = 1;
				switch (attr) {
					case 0:
						xga = 4;
						break; //sum, see uk+
					case 1:
						xga = 1;
						break; //count, see cm
					case 2:
						xga = 8;
						break; //mean
					case 3:
						xga = 16;
						break; //cm2
					case 4:
						xga = 31;
						break; //cm3
					case 5:
						xga = 51;
						break; //cm4
					case 6:
						xga = 16;
						break; //variance
				}
				return 2 * input.getM() + xga * input.getM(); //scan for min/max, groupedagg

			}
			case "rmempty": {
				HashMap<String, String> paramsMap = inst.getParameterMap();
				int attr = paramsMap.get("margin").equals("rows") ? 0 : 1;
				switch (attr) {
					case 0: //remove rows
						// TODO: Copied from old implementation but maybe reverse the cases?
						return ((input.isSparse()) ? input.getM() : input.getM() * Math.ceil(1.0d / input.getS()) / 2) +
								DEFAULT_NFLOP_CP * output.getCells();
					case 1: //remove cols
						return input.getN() * Math.ceil(1.0d / input.getS()) / 2 +
								DEFAULT_NFLOP_CP * output.getCells();
					default:
						throw new DMLRuntimeException("Invalid margin type for opcode " + opcode + ".");
				}
			}
			default:
				throw new DMLRuntimeException("Estimation for instruction " + inst.getCPInstructionType() + " with op. code '" + opcode +"' is not supported yet.");
		}
	}

	public boolean constructSparkJob(Instruction inst) {
		// return true if the parsed instruction would trigger Job execution
		SPInstruction sourceInstruction = _transformations.peekLast();
		if (sourceInstruction == null && inst instanceof CPInstruction) { // no spark job in construction
			return false;
		} else if (sourceInstruction == null /* && inst instanceof CPInstruction */) { // start constructing new spark job
			_transformations.add((SPInstruction) (inst));
			return false;
		} else if (/* sourceInstruction != null && */ inst instanceof CPInstruction) {
			if (inst instanceof VariableCPInstruction) {
				return false; // job construction may continue
			} else {
				return true; // job construction end detected
			}
		}
		/* else = else if (sourceInstruction != null && inst instanceof SPInstruction) */
		// TODO: simplify if first else if and this statements stay the same
		_transformations.add((SPInstruction) (inst));
		return false;
	}

	public double getTimeEstimateSparkJob() {
		// TODO: implement
		// clear the list to allow future job constructions
		_transformations.clear();
		return 0;
	}

	/**
	 * Intended to be used to get the NFLOP for SPInstructions.
	 * 'parse' because the cost of each instruction is to be
	 * collected and the cost is to be computed at the end based on
	 * all Spark instructions TODO: change
	 * @param inst
	 * @return
	 */
	protected double getTimeEstimateSPInst(SPInstruction inst) {
		// NOTE: all factors and logic are dummy for now
		// declare resource-dependant metrics
		double localCost = 0;  // [nflop] cost for computing executed in executors
		double globalCost = 0;  // [nflop] cost for computing executed in driver
		double IOCost = 0; // [s] cost for shuffling data and writing/reading to HDFS/S3
		// TODO: consider the case of matrix with dims=1
		// NOTE: consider if is needed to include the cost for final aggregation within the Spark Driver (CP)
		if (inst instanceof ReblockSPInstruction || inst instanceof CSVReblockSPInstruction) {
			int complexityFactor = (inst instanceof ReblockSPInstruction) ? 1 : 2;
			VarStats output = getStats(((UnarySPInstruction) inst).getOutputVariableName());
			// for reblock instructions the rdd stats are initiated first here
			output.rddStats = new RDDStats(output);
			output.rddStats.loadCharacteristics();
			return (double) (complexityFactor * output.getCells()) / output.rddStats.numParallelTasks;
		} else if (inst instanceof CheckpointSPInstruction) {
			VarStats output = getStats(((CheckpointSPInstruction) inst).getOutputVariableName());
			// for reblock instructions the rdd stats are initiated first here
			output.rddStats = new RDDStats(output);
			output.rddStats.loadCharacteristics();
			// no compute or load cost related to checkpointing
			return 0;
		} else if (inst instanceof RandSPInstruction) {
			RandSPInstruction randInst = (RandSPInstruction) inst;
			String opcode = randInst.getOpcode();
			VarStats output = getStats(randInst.output.getName());
			// update sparsity here
			output.characteristics.setNonZeros((long) (output.getCells()*randInst.getSparsity()));
			RDDStats outputRDD = new RDDStats(output);

			int complexityFactor = 1;
			switch (opcode.toLowerCase()) {
				case DataGen.RAND_OPCODE:
					complexityFactor = 32; // higher complexity for random number generation
				case DataGen.SEQ_OPCODE:
					// first op. from the new stage: parallelize/read from scratch file
					globalCost += complexityFactor*outputRDD.numBlocks; // cp random number generation
					if (outputRDD.numBlocks < RandSPInstruction.INMEMORY_NUMBLOCKS_THRESHOLD) {
						long parBlockSize = MatrixBlock.estimateSizeDenseInMemory(outputRDD.numBlocks, 1);
						IOCost += IOCostUtils.getSparkTransmissionCost(parBlockSize, outputRDD.numParallelTasks);
					} else {
						IOCost += IOCostUtils.getWriteTime(outputRDD.numBlocks, 1, 1.0, HDFS_SOURCE_IDENTIFIER, TEXT); // driver writes down
						IOCost += IOCostUtils.getReadTime(outputRDD.numBlocks, 1, 1.0, HDFS_SOURCE_IDENTIFIER, TEXT); // executors read
						localCost += outputRDD.numBlocks; // mapToPair cost
					}
					localCost += complexityFactor*outputRDD.cpStats.getCells(); // mapToPair cost
					output.rddStats = outputRDD;
					return globalCost;
				case DataGen.SAMPLE_OPCODE:
					// first op. from the new stage: parallelize
					complexityFactor = 32; // TODO: set realistic factor
					globalCost += complexityFactor*outputRDD.numPartitions; // cp random number generation
					long parBlockSize = MatrixBlock.estimateSizeDenseInMemory(outputRDD.numPartitions, 1);
					IOCost += IOCostUtils.getSparkTransmissionCost(parBlockSize, outputRDD.numParallelTasks);
					localCost += outputRDD.numBlocks; // flatMap cost
					localCost += complexityFactor*outputRDD.numBlocks; // mapToPairCost cost
					// sortByKey -> new stage
					long randBlockSize = MatrixBlock.estimateSizeDenseInMemory(outputRDD.numBlocks, 1);
					IOCost += IOCostUtils.getShuffleCost(randBlockSize, outputRDD.numParallelTasks);
					localCost += outputRDD.cpStats.getCells();
					// sortByKey -> shuffling?
				case DataGen.FRAME_OPCODE:
			}
			return globalCost;
		} else if (inst instanceof ComputationSPInstruction) {
			VarStats input = getStats(((ComputationSPInstruction) inst).input1.getName());
			getRDDHandleAndEstimateTime(input);
			VarStats output = getStats(((ComputationSPInstruction) inst).output.getName());
			output.rddStats.loadCharacteristics();
			return (double) output.getCells() / output.rddStats.numParallelTasks;
		}

		throw new DMLRuntimeException("Unsupported instruction: " + inst.getOpcode());
	}

	/**
	 * Intended to handle RDDStats retrievals so the I/O
	 * can be computed correctly. This method should also
	 * reserve the necessary memory for the RDD on the executors.
	 * @param input input statistics
	 * @return
	 */
	private double getRDDHandleAndEstimateTime(VarStats input) {
		double ret = 0;
		if (input.rddStats == null) {
			throw new RuntimeException("The urrent RDDStats should have been already initiated");
		}
		RDDStats inputRDD = input.rddStats;
		if (inputRDD.distributedMemory < 0) {
			if (input.allocatedMemory >= 0) { // dirty or cached
				if (!_parRDDs.reserve(inputRDD.distributedMemory)) {
					if (input.isDirty) {
						ret += IOCostUtils.getWriteTime(input.getM(), input.getN(), input.getS(), HDFS_SOURCE_IDENTIFIER, BINARY);
						// TODO: think when to set it to true
						input.isDirty = false;
					}
					inputRDD.loadCharacteristics(); // possibly redundant
					ret += IOCostUtils.getReadTime(input.getM(), input.getN(), input.getS(), HDFS_SOURCE_IDENTIFIER, BINARY) / inputRDD.numParallelTasks;
				} else {
					inputRDD.loadCharacteristics();    // possibly redundant
					ret += IOCostUtils.getSparkTransmissionCost(inputRDD.distributedMemory, inputRDD.numParallelTasks);
				}
			} else { // on hdfs
				if (input.fileInfo == null || input.fileInfo.length != 2)
					throw new DMLRuntimeException("File info missing for a file to be read on Spark.");
				inputRDD.loadCharacteristics(); // possibly redundant
				ret += IOCostUtils.getReadTime(input.getM(), input.getN(), input.getS(), (String) input.fileInfo[0], (FileFormat) input.fileInfo[1]) / inputRDD.numParallelTasks;
				input.isDirty = false; // possibly redundant
			}
		}
		// if RDD handle exists -> no additional cost to add
		return ret;
	}

	/////////////////////
	// I/O Costs       //
	/////////////////////

	private double getLoadTime(VarStats input) throws CostEstimationException {
		if (input.isScalar() || input.allocatedMemory > 0) return 0.0;
		// loading from RDD
		if (input.rddStats != null) { // input.rddStats == null also for var output of reblock/checkpoint instructions
			double computeAndLoadTime;
			computeAndLoadTime = getTimeEstimateSparkJob();
			if (OptimizerUtils.checkSparkCollectMemoryBudget(input.characteristics, usedMememory, false)) { // .collect()
				long sizeEstimate = OptimizerUtils.estimatePartitionedSizeExactSparsity(input.characteristics);
				computeAndLoadTime += IOCostUtils.getSparkTransmissionCost(sizeEstimate, input.rddStats.numParallelTasks);
			} else { // redirect through HDFS
				computeAndLoadTime += IOCostUtils.getWriteTime(input.getM(), input.getN(), input.getS(), HDFS_SOURCE_IDENTIFIER, BINARY) / input.rddStats.numParallelTasks +
						IOCostUtils.getReadTime(input.getM(), input.getN(), input.getS(), HDFS_SOURCE_IDENTIFIER, BINARY); // cost for writting to HDFS on executors and reading back on driver
			}
			putInMemory(input);
			return computeAndLoadTime;
		}
		// loading from a file
		if (input.fileInfo == null || input.fileInfo.length != 2) {
			throw new DMLRuntimeException("Time estimation is not possible without file info.");
		}
		else if (!input.fileInfo[0].equals(HDFS_SOURCE_IDENTIFIER) && !input.fileInfo[0].equals(S3_SOURCE_IDENTIFIER)) {
			throw new DMLRuntimeException("Time estimation is not possible for data source: "+ input.fileInfo[0]);
		}
		putInMemory(input);
		return IOCostUtils.getReadTime(input.getM(), input.getN(), input.getS(), (String) input.fileInfo[0], (FileFormat) input.fileInfo[1]);
	}

	private void putInMemory(VarStats input) throws CostEstimationException {
		if (input.isScalar()) return;
		long sizeEstimate = OptimizerUtils.estimateSize(input.characteristics);
		if (sizeEstimate + usedMememory > localMemory)
			throw new CostEstimationException("Insufficient local memory");
		usedMememory += sizeEstimate;
		input.allocatedMemory = sizeEstimate;
	}

	private void removeFromMemory(VarStats input) {
		if (input == null) return; // scalars or variables never put in memory
		usedMememory -= input.allocatedMemory;
		input.allocatedMemory = -1;
	}

	/////////////////////
	// HELPERS         //
	/////////////////////

	private static int getComputationFactorUAOp(String opcode) {
		switch (opcode) {
			case "uatrace": case "uaktrace":
				return 2;
			case "uak+": case "uark+": case "uack+":
				return 4; // 1*k+
			case "uasqk+": case "uarsqk+": case "uacsqk+":
				return 5; // +1 for multiplication to square term
			case "uamean": case "uarmean": case "uacmean":
				return 7; // 1*k+
			case "uavar": case "uarvar": case "uacvar":
				return 14;
			default:
				return 1;
		}
	}

	// NOTE: use for ideas only
//	protected double getTimeEstimateSPInst(SPInstruction inst) {
//		// declare resource-dependant metrics
//		double localCost = 0;  // [nflop] cost for computing executed in executors
//		double globalCost = 0;  // [nflop] cost for computing executed in driver
//		double IOCost = 0; // [s] cost for shuffling data and writing/reading to HDFS/S3
//		// TODO: consider the case of matrix with dims=1
//		// NOTE: consider if is needed to include the cost for final aggregation within the Spark Driver (CP)
//		if (inst instanceof AggregateTernarySPInstruction) {
//			// TODO: need to have a way to associate mVars from _stats with a
//			// 	potentially existing virtual PairRDD - MatrixObject
//			// NOTE: leave it for later once I figure out how to do it for unary instructions
//		} else if (inst instanceof AggregateUnarySPInstruction) {
//			AggregateUnarySPInstruction currentInst = (AggregateUnarySPInstruction) inst;
//			String opcode = currentInst.getOpcode();
//			AggBinaryOp.SparkAggType aggType = currentInst.getAggType();
//			AggregateUnaryOperator op = (AggregateUnaryOperator) currentInst.getOperator();
//			VarStats input = getStats(currentInst.input1.getName());
//			RDDStats inputRDD = input.rddStats;
//			RDDStats currentRDD = inputRDD;
//			VarStats outputStats = getStats(currentInst.output.getName());
//
//			int k = getComputationFactorUAOp(opcode);
//			// TODO: RRDstats extra required to keep at least the number of
//			// 	blocks that each next operator operates on: e.g. filter (and mapByKey) is reducing this number,
//			// 	probably better to create and store only intermediate RDDstats shared between instructions
//			// 	since only these are needed for retrieving only intra instructions
//			// TODO: later think of how to handle getting null for stats
//			if (inputRDD == null) {
//				throw new DMLRuntimeException("RDD stats should have been already initiated");
//			}
//			if (opcode.equals("uaktrace")) {
//				// add cost for filter op
//				localCost += currentRDD.numBlocks;
//				currentRDD = RDDStats.transformNumBlocks(currentRDD, currentRDD.rlen); // only the diagonal blocks left
//			}
//			if (aggType == AggBinaryOp.SparkAggType.SINGLE_BLOCK) {
//				if (op.sparseSafe) {
//					localCost += currentRDD.numBlocks; // filter cost
//					// TODO: decide how to reduce numBlocks
//				}
//				localCost += k*currentRDD.numValues*currentRDD.sparsity; // map cost
//				// next op is fold -> end of the current Job
//				// end of Job -> no need to assign the currentRDD to the output (output is no RDD)
//				localCost += currentRDD.numBlocks; // local folding cost
//				// TODO: shuffle cost to bring all pairs to the driver (CP)
//				// NOTE: neglect the added CP compute cost for folding the distributed aggregates
//			} else if (aggType == AggBinaryOp.SparkAggType.MULTI_BLOCK){
//				localCost += k*currentRDD.numValues*currentRDD.sparsity; // mapToPair cost
//				// NOTE: the new unique number of keys should be
//				// next op is combineByKey -> new stage
//				localCost += currentRDD.numBlocks + currentRDD.numPartitions; // local merging * merging partitions
//				if (op.aggOp.existsCorrection())
//					localCost += currentRDD.numBlocks; // mapValues cost for the correction
//			} else {  // aggType == AggBinaryOp.SparkAggType.NONE
//				localCost += k*currentRDD.numValues*currentRDD.sparsity;
//				// no reshuffling -> inst is packed with the next spark operation
//			}
//
//			return globalCost;
//		} else if (inst instanceof RandSPInstruction) {
//			RandSPInstruction randInst = (RandSPInstruction) inst;
//			String opcode = randInst.getOpcode();
//			VarStats output = getStats(randInst.output.getName());
//			// update sparsity here
//			output.characteristics.setNonZeros((long) (output.getCells()*randInst.getSparsity()));
//			RDDStats outputRDD = new RDDStats(output);
//
//			int complexityFactor = 1;
//			switch (opcode.toLowerCase()) {
//				case DataGen.RAND_OPCODE:
//					complexityFactor = 32; // higher complexity for random number generation
//				case DataGen.SEQ_OPCODE:
//					// first op. from the new stage: parallelize/read from scratch file
//					globalCost += complexityFactor*outputRDD.numBlocks; // cp random number generation
//					if (outputRDD.numBlocks < RandSPInstruction.INMEMORY_NUMBLOCKS_THRESHOLD) {
//						long parBlockSize = MatrixBlock.estimateSizeDenseInMemory(outputRDD.numBlocks, 1);
//						IOCost += IOCostUtils.getSparkTransmissionCost(parBlockSize, outputRDD.numParallelTasks);
//					} else {
//						IOCost += IOCostUtils.getWriteTime(outputRDD.numBlocks, 1, 1.0, HDFS_SOURCE_IDENTIFIER, TEXT); // driver writes down
//						IOCost += IOCostUtils.getReadTime(outputRDD.numBlocks, 1, 1.0, HDFS_SOURCE_IDENTIFIER, TEXT); // executors read
//						localCost += outputRDD.numBlocks; // mapToPair cost
//					}
//					localCost += complexityFactor*outputRDD.numValues; // mapToPair cost
//					output.rddStats = outputRDD;
//					return globalCost;
//				case DataGen.SAMPLE_OPCODE:
//					// first op. from the new stage: parallelize
//					complexityFactor = 32; // TODO: set realistic factor
//					globalCost += complexityFactor*outputRDD.numPartitions; // cp random number generation
//					long parBlockSize = MatrixBlock.estimateSizeDenseInMemory(outputRDD.numPartitions, 1);
//					IOCost += IOCostUtils.getSparkTransmissionCost(parBlockSize, outputRDD.numParallelTasks);
//					localCost += outputRDD.numBlocks; // flatMap cost
//					localCost += complexityFactor*outputRDD.numBlocks; // mapToPairCost cost
//					// sortByKey -> new stage
//					long randBlockSize = MatrixBlock.estimateSizeDenseInMemory(outputRDD.numBlocks, 1);
//					IOCost += IOCostUtils.getShuffleCost(randBlockSize, outputRDD.numParallelTasks);
//					localCost += outputRDD.numValues;
//					// sortByKey -> shuffling?
//				case DataGen.FRAME_OPCODE:
//			}
//			return globalCost;
//		}
//
//		throw new DMLRuntimeException("Unsupported instruction: " + inst.getOpcode());
//	}
}
