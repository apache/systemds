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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.*;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.instructions.spark.*;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

import static org.apache.sysds.lops.Data.PREAD_PREFIX;
import static org.apache.sysds.lops.DataGen.*;
import static org.apache.sysds.resource.cost.IOCostUtils.*;
import static org.apache.sysds.resource.cost.SparkCostUtils.getRandInstTime;

import java.util.*;

/**
 * Class for estimating the execution time of a program.
 * For estimating the time for new set of resources,
 * a new instance of CostEstimator should be created.
 */
public class CostEstimator
{
	// protected static final Log LOG = LogFactory.getLog(CostEstimator.class.getName());
	private static final int DEFAULT_NUM_ITER = 15;

	// Non-static members
	private final Program _program;
	private final IOCostUtils.IOMetrics driverMetrics;
	private final IOCostUtils.IOMetrics executorMetrics;
	private final SparkExecutionContext.MemoryManagerParRDDs _parRDDs;
	// declare here the hashmaps
	private final HashMap<String, VarStats> _stats;
	private LinkedList<VarStats> _lineage;
	private final HashSet<String> _functions;
	private final long localMemoryLimit; // refers to the drivers JVM memory
	private long freeLocalMemory;
	private final long distributedMemoryLimit; // refers to the combined Spark JVM memory
	private long freeDistributedMemory;

	/**
	 * Entry point for estimating the execution time of a program.
	 * @param program compiled runtime program
	 * @return estimated time for execution of the program
	 * given the resources set in {@link SparkExecutionContext}
	 * @throws CostEstimationException in case of errors
	 */
	public static double estimateExecutionTime(Program program, CloudInstance driverNode, CloudInstance executorNode) throws CostEstimationException {
		CostEstimator estimator = new CostEstimator(program, driverNode, executorNode);
		double costs = estimator.getTimeEstimate();
		return costs;
	}
	public CostEstimator(Program program, CloudInstance driverNode, CloudInstance executorNode) {
		_program = program;
		driverMetrics = new IOCostUtils.IOMetrics(driverNode);
		executorMetrics = executorNode != null? new IOCostUtils.IOMetrics(executorNode) : null;
		// initialize here the hashmaps
		_stats = new HashMap<>();
		_lineage = new LinkedList<>();
		_functions = new HashSet<>();
		localMemoryLimit = (long) OptimizerUtils.getLocalMemBudget();
		freeLocalMemory = localMemoryLimit;
		// obtain the whole available memory budget
		distributedMemoryLimit = executorNode != null? (long) SparkExecutionContext.getDataMemoryBudget(false, false) : -1;
		freeDistributedMemory = distributedMemoryLimit;
		this._parRDDs = new SparkExecutionContext.MemoryManagerParRDDs(0.1);
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
			result = new VarStats(statsName, null);
		}
		return result;
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
			ret *= DEFAULT_NUM_ITER;
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
			ret *= OptimizerUtils.getNumIterations(tmp, DEFAULT_NUM_ITER);
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
				String argName = finst.getFunArgNames().get(i);
				VarStats argStats = getStats(inputs[i].getName());
				if (inputs[i].getName().equals(argName)) {
					if (argStats != _stats.get(argName))
						throw  new RuntimeException("Overriding referenced variable within a function call is not a handled case");
					// reference duplication in different domain
					argStats.selfRefCount++;
				} else {
					// passing the reference to another variable
					argStats.refCount++;
					_stats.put(finst.getFunArgNames().get(i), argStats);
				}
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
	public void maintainStats(Instruction inst) {
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
					VarStats varStats = new VarStats(varName, dataCharacteristics);
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
					//
					outputStats = getStats(varName);
					_stats.put(vinst.getInput2().getName(), outputStats);
					outputStats.refCount++;
					break;
				case "mvvar":
					VarStats statsToMove = _stats.remove(varName);
					String newName = vinst.getInput2().getName();
					if (statsToMove != null) statsToMove.varName = newName;
					_stats.put(newName, statsToMove);
					break;
				case "rmvar":
					for (CPOperand inputOperand: vinst.getInputs()) {
						VarStats inputVar = _stats.remove(inputOperand.getName());
						if (inputVar == null) continue; // inputVar == null for scalars
						// actually remove from memory only if not referenced more than once
						if (--inputVar.selfRefCount > 0) {
							_stats.put(inputOperand.getName(), inputVar);
						} else if (--inputVar.refCount < 1) {
							removeFromMemory(inputVar);
						}
					}
					break;
				case "castdts":
					VarStats scalarStats = new VarStats(vinst.getOutputVariableName(), null);
					_stats.put(vinst.getOutputVariableName(), scalarStats);
					break;
				case "write":
					String fileName = vinst.getInput2().isLiteral()? vinst.getInput2().getLiteral().getStringValue() : "hdfs_file";
					String dataSource = IOCostUtils.getDataSource(fileName);
					String formatString = vinst.getInput3().getLiteral().getStringValue();
					_stats.get(varName).fileInfo = new Object[] {dataSource, FileFormat.safeValueOf(formatString)};
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
		} else if (inst instanceof AggregateUnaryCPInstruction) {
			// specific case to aid future dimensions inferring
			String opcode = inst.getOpcode();
			if (!(opcode.equals("nrow") || opcode.equals("ncol") || opcode.equals("length"))) {
				return;
			}
			AggregateUnaryCPInstruction auinst = (AggregateUnaryCPInstruction) inst;
			VarStats inputStats = getStats(auinst.input1.getName());
			String outputName = auinst.getOutputVariableName();
			VarStats outputStats;
			if (opcode.equals("nrow")) {
				if (inputStats.getM() < 0) return;
				outputStats = new VarStats(String.valueOf(inputStats.getM()), null);
			} else if (opcode.equals("ncol")) {
				if (inputStats.getN() < 0) return;
				outputStats = new VarStats(String.valueOf(inputStats.getN()), null);
			} else { // if (opcode.equals("length"))
				if (inputStats.getCells() < 0) return;
				outputStats = new VarStats(String.valueOf(inputStats.getCells()), null);
			}
			_stats.put(outputName, outputStats);
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
			// handle special cases
			if (inst instanceof ReblockSPInstruction || inst instanceof CSVReblockSPInstruction) {
				VarStats input = getStats(((UnarySPInstruction) inst).input1.getName());
				if (input.allocatedMemory < 0 && input.fileInfo == null) {
					throw new RuntimeException("Reblock instruction allowed only loaded variables or for such with given source file");
				}
				// re-assigning to null is fine as long as the allocated memory is non-negative
				output.fileInfo = input.fileInfo;
				output.allocatedMemory = input.allocatedMemory;
			} else if (inst instanceof CheckpointSPInstruction) {
				VarStats input = getStats(((CheckpointSPInstruction) inst).input1.getName());
				// re-assigning to null is fine: checkpoint for objects generated by Spark
				output.fileInfo = input.fileInfo;
				output.allocatedMemory = input.allocatedMemory;
			}else if (inst instanceof RandSPInstruction) {
				// update sparsity here
				long nnz = (long) (output.getCells()*((RandSPInstruction) inst).getSparsity());
				output.characteristics.setNonZeros(nnz);
			}
			// init rdd stats object (distributedSize and numPartitions are initialized separately)
			output.rddStats = new RDDStats(output);
		}
	}

	public double getTimeEstimateInst(Instruction inst) throws CostEstimationException {
		double ret = 0;
		if (inst instanceof CPInstruction) {
			ret = getTimeEstimateCPInst((CPInstruction)inst);
		} else { // inst instanceof SPInstruction
			parseSPInst((SPInstruction) inst);
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
	 * @param inst instruction for estimation
	 * @return estimated time in seconds
	 */
	public double getTimeEstimateCPInst(CPInstruction inst) throws CostEstimationException {
		double time = 0;
		VarStats output = null;
		if (inst instanceof VariableCPInstruction) {
			String opcode = inst.getOpcode();
			VariableCPInstruction vinst = (VariableCPInstruction) inst;
			VarStats input = null;
			if (opcode.startsWith("cast")) {
 				input = getStatsWithDefaultScalar(vinst.getInput1().getName());
				output = getStatsWithDefaultScalar(vinst.getOutput().getName());
				CPCostUtils.assignOutputMemoryStats(inst, output, input);
			}
			else if (opcode.equals("write")) {
				input = getStatsWithDefaultScalar(vinst.getInput1().getName());
				time += IOCostUtils.getDiskWriteTime(input, driverMetrics); // I/O estimate
			}
			time += input == null? 0 : loadCPVarStatsAndEstimateTime(input);
			time += CPCostUtils.getVariableInstTime(vinst, input, output, driverMetrics);
		}
		else if (inst instanceof UnaryCPInstruction) {
			UnaryCPInstruction uinst = (UnaryCPInstruction) inst;
			output = getStatsWithDefaultScalar(uinst.getOutput().getName());
			if (inst instanceof DataGenCPInstruction || inst instanceof StringInitCPInstruction) {
				String[] s = InstructionUtils.getInstructionParts(uinst.getInstructionString());
				VarStats rows = getStatsWithDefaultScalar(s[1]);
				VarStats cols = getStatsWithDefaultScalar(s[2]);
				CPCostUtils.assignOutputMemoryStats(inst, output, rows, cols);
				time += CPCostUtils.getDataGenCPInstTime(uinst, output, driverMetrics);
			} else {
				// UnaryCPInstruction input can be any type of object
				VarStats input = getStatsWithDefaultScalar(uinst.input1.getName());
				// a few of the unary instructions take second optional argument of type matrix
				VarStats weights = uinst.input2 == null ? null : getStats(uinst.input2.getName());

				if (inst instanceof IndexingCPInstruction) {
					IndexingCPInstruction ixdinst = (IndexingCPInstruction) inst;
					VarStats rowLower = getStatsWithDefaultScalar(ixdinst.getRowLower().getName());
					VarStats rowUpper = getStatsWithDefaultScalar(ixdinst.getRowUpper().getName());
					VarStats colLower = getStatsWithDefaultScalar(ixdinst.getColLower().getName());
					VarStats colUpper = getStatsWithDefaultScalar(ixdinst.getColUpper().getName());
					CPCostUtils.assignOutputMemoryStats(inst, output, input, rowLower, rowUpper, colLower, colUpper);
				} else {
					CPCostUtils.assignOutputMemoryStats(inst, output, input);
				}

				time += loadCPVarStatsAndEstimateTime(input);
				time += weights == null ? 0 : loadCPVarStatsAndEstimateTime(weights);
				time += CPCostUtils.getUnaryInstTime(uinst, input, weights, output, driverMetrics);
			}
		}
		else if (inst instanceof BinaryCPInstruction) {
			BinaryCPInstruction binst = (BinaryCPInstruction) inst;
			VarStats input1 = getStatsWithDefaultScalar(binst.input1.getName());
			VarStats input2 = getStatsWithDefaultScalar(binst.input2.getName());
			VarStats weights = binst.input3 == null? null : getStatsWithDefaultScalar(binst.input3.getName());
			output = getStatsWithDefaultScalar(binst.output.getName());
			CPCostUtils.assignOutputMemoryStats(inst, output, input1, input2);

			time += loadCPVarStatsAndEstimateTime(input1);
			time += loadCPVarStatsAndEstimateTime(input2);
			time += weights == null? 0 : loadCPVarStatsAndEstimateTime(weights);
			time += CPCostUtils.getBinaryInstTime(binst, input1, input2, weights, output, driverMetrics);

		}
		else if (inst instanceof ParameterizedBuiltinCPInstruction) {
			if (inst instanceof ParamservBuiltinCPInstruction) {
				throw new RuntimeException("ParamservBuiltinCPInstruction is not supported for estimation");
			}
			ParameterizedBuiltinCPInstruction pinst = (ParameterizedBuiltinCPInstruction) inst;
			String inputName = pinst.getParam("target");
			if (inputName == null) {
				throw new RuntimeException("ParameterizedBuiltinCPInstruction without given target object is not supported for estimation");
			}
			VarStats input = getStatsWithDefaultScalar(inputName);
			output = getStatsWithDefaultScalar(pinst.getOutputVariableName());
			CPCostUtils.assignOutputMemoryStats(inst, output, input);

			time += loadCPVarStatsAndEstimateTime(input);
			time += CPCostUtils.getParameterizedBuiltinInstTime(pinst, input, output, driverMetrics);
		} else if (inst instanceof MultiReturnBuiltinCPInstruction) {
			MultiReturnBuiltinCPInstruction mrbinst = (MultiReturnBuiltinCPInstruction) inst;
			VarStats input = getStats(mrbinst.input1.getName());
			VarStats[] outputs = new VarStats[mrbinst.getOutputs().size()];
			int i = 0;
			for (CPOperand operand : mrbinst.getInputs()) {
				if (!operand.isMatrix()) {
					throw new DMLRuntimeException("MultiReturnBuiltinCPInstruction expects only matrix output objects");
				}
				VarStats current = getStats(operand.getName());
				outputs[i] = current;
				// consider moving out the next two fcalls outside the loop
				CPCostUtils.assignOutputMemoryStats(inst, current, input);
				putInMemory(current);
				i++;
			}
			time += loadCPVarStatsAndEstimateTime(input);
			time += CPCostUtils.getMultiReturnBuiltinInstTime(mrbinst, input, outputs, driverMetrics);
			// the only place to return directly here (output put in memory already)
			return time;
		}
		else if (inst instanceof ComputationCPInstruction) {
			if (inst instanceof MultiReturnParameterizedBuiltinCPInstruction || inst instanceof CompressionCPInstruction || inst instanceof DeCompressionCPInstruction) {
				throw new RuntimeException(inst.getClass().getName() + " is not supported for estimation");
			}
			ComputationCPInstruction cinst = (ComputationCPInstruction) inst;
			VarStats input1 = getStatsWithDefaultScalar(cinst.input1.getName()); // 1 input: AggregateTernaryCPInstruction
			// in general only the first input operand is guaranteed initialized
			// assume they can be also scalars (often operands are some literal or scalar arguments not related to the cost estimation)
			VarStats input2 = cinst.input2 == null? null : getStatsWithDefaultScalar(cinst.input2.getName()); // 2 inputs: CtableCPInstruction, PMMJCPInstruction
			VarStats input3 = cinst.input3 == null? null : getStatsWithDefaultScalar(cinst.input3.getName()); // 3 inputs: TernaryCPInstruction, CtableCPInstruction
			VarStats input4 = cinst.input4 == null? null : getStatsWithDefaultScalar(cinst.input4.getName()); // 4 inputs (possibly): QuaternaryCPInstruction
			output = getStatsWithDefaultScalar(cinst.getOutput().getName());
			CPCostUtils.assignOutputMemoryStats(inst, output, input1, input2, input3, input4);

			time += loadCPVarStatsAndEstimateTime(input1);
			time += input2 == null? 0 : loadCPVarStatsAndEstimateTime(input2);
			time += input3 == null? 0 : loadCPVarStatsAndEstimateTime(input3);
			time += input4 == null? 0 : loadCPVarStatsAndEstimateTime(input4);
			time += CPCostUtils.getComputationInstTime(cinst, input1, input2, input3, input4, output, driverMetrics);
		}
		else if (inst instanceof BuiltinNaryCPInstruction) {
			BuiltinNaryCPInstruction bninst = (BuiltinNaryCPInstruction) inst;
			output = getStatsWithDefaultScalar(bninst.getOutput().getName());
			// putInMemory(output);
			if (bninst instanceof ScalarBuiltinNaryCPInstruction) {
				return CPCostUtils.getBuiltinNaryInstTime(bninst, null, output, driverMetrics);
			}
			VarStats[] inputs = new VarStats[bninst.getInputs().length];
			int i = 0;
			for (CPOperand operand : bninst.getInputs()) {
				if (operand.isMatrix()) {
					VarStats input = getStatsWithDefaultScalar(operand.getName());
					time += loadCPVarStatsAndEstimateTime(input);
					inputs[i] = input;
					i++;
				}
			}
			// trim the arrays to its actual size
			inputs = Arrays.copyOf(inputs, i + 1);
			CPCostUtils.assignOutputMemoryStats(inst, output, inputs);
			time += CPCostUtils.getBuiltinNaryInstTime(bninst, inputs, output, driverMetrics);
		}
		else { // SqlCPInstruction
			throw new RuntimeException(inst.getClass().getName() + " is not supported for estimation");
		}

		if (output != null) putInMemory(output);
		return time;
	}

	public double getTimeEstimateSparkJob(VarStats varToCollect) throws CostEstimationException {
		/* Explanation about the logic:
		 * Here is assumed executing a single Job at a time on the whole cluster. note: maybe not valid parfors?
		 * The execution time (latency) of each Spark job can be computed as a sum of the latency for all the stages.
		 * The latency for each stage can be computed simplified as sum of the execution time for all executed tasks
		 * divided (with ceiling) by the number of available execution slots across the whole cluster.
		 * A tasks within a single stage executes a single Spark transformation on a single partition of a
		 * distributed dataset (RDD).
		 *
		 * The latency of tasks executing the same transformation on a different partition of the dataset
		 * can be different due to difference in the data complexity regarding the current transformation
		 * or simply due to current differences of the available underlying resources of the different executors.
		 * This, however, can be hardly taken into account et static estimation of the execution time of a Spark job.
		 *
		 * Each transformation has potentially a different computing complexity
		 * and/or operates on a different dataset - output RDD of the previous one.
		 * By knowing potentially scheduled Spark tasks by the SystemDS control program (comprising the Spark driver),
		 * the general latency of each different transformation can be statically estimated when the
		 * data characteristics of relevant RDDs are known or can be inferred by the SystemDS compiler.
		 *
		 * The task latency can be defined by several metrics which represent the general steps of the execution of each tasks
		 * and the sum of all these metrics delivers the total execution time of that task.
		 * Here are the metrics which their relevance for time estimation:
		 * 	- Scheduler Delay: time waiting for available slot for execution - should ne minimal since SystemDS will not submit several jobs at a time
		 * 	- Task Deserialization Time: Time to deserialize the task logic into executable bytecode
		 * 	- Shuffle Read Time: Time for reading total shuffle bytes and records, includes both data read locally and data read from remote executors
		 * 	- Executor Computing Time: Pure computation time for the transformation
		 * 	- Shuffle Write Time: Time for writing bytes and records to disk in order to be read by a shuffle in a future stage
		 * 	- Result Serialization Time: Time is the time taken to serialize the result of a task before sending it back to the driver
		 * 	- Getting Result Time:
		 *
		 *
		 * Due to the logic of some spark instruction (in SystemDS) not all data characteristics an be inferred,
		 * e.g. sparsity or output dimensions, which does not allow statically inferring all single Spark transformations
		 * that would be scheduled for each spark instructions, but a approximation spark plan is statically created in {@code SparkCostUtils}.
		 *
		 * Another complication at mapping the Spark instructions to spark tasks for estimating the final
		 * stage latency and by that the job latency, is that each instruction can be comprised of
		 * one or several transformations which being narrow or wide. That means that one instruction
		 * can define only a part of a single spark stage or to define more then one stages.
		 *
		 * The approach to handle the explained challenges is to iterate over all
		 * Spark instructions, defining a Spark job, and for each of them
		 * the execution time is estimated separately following these rules:
		 * 	- if the instruction contains only narrow transformations, estimate computation time only
		 *	- if the instruction contains also wide transformations, estimate and add shuffle time for each of them
		 * 	- if the instruction requires reading a file, estimate and add read time
		 * 	- for WriteSPInstruction, estimate only writing time
		 * 	- if the instruction contains an explicit action, estimate and add transfer time + init. VarStats.availableMemory
		 * 	- time for non explicit actions is handled by  getLoadTime() once the RDD object is needed for a CP instruction
		 *
		 * computation time is defined as: max(mem. scan time, processing time) + mem. write time;
		 * shuffle time is defined as: write time + read time, where the read time is always accounted
		 * together with the write time to allow next instruction to assume that this operation is already estimated;
		 */
		LinkedList<VarStats> newLineage = new LinkedList<>();
		double time = -1;
		for (VarStats current: _lineage) {
			RDDStats currentRdd = current.rddStats;
			if (current.equals(varToCollect)) {
				if (currentRdd.isCollected) {
					putInMemory(current);
				}
				// TODO: consider is this is the best thing to do here
				time = currentRdd.cost;
				currentRdd.isCollected = true;
				// once collected, the objected is persisted in the CP memory
				currentRdd.cost = 0;
			} else {
				if (current.refCount > 0) {
					newLineage.add(current);
				}
			}
		}
		if (time <= 0) {
			throw new RuntimeException("The estimated Job execution time should be always positive");
		}
		// clear the old list but retain still referenced variables;
		// the retained variables carry the cost (estimated time)
		// for their generation and by that represent the cost lineage;
		_lineage = newLineage;
		return time;
	}

	/**
	 * Intended to be used to get the NFLOP for SPInstructions.
	 * 'parse' because the cost of each instruction is to be
	 * collected and the cost is to be computed at the end based on
	 * all Spark instructions TODO: change
	 * @param inst
	 * @return
	 */
	public void parseSPInst(SPInstruction inst) throws CostEstimationException {
		/* Logic for the parallelization factors:
		 * the given executor metrics relate to peak performance per node,
		 * utilizing all the resources available, but the Spark operations
		 * are executed by several tasks per node so the execution/read time
		 * per operation is the potential execution time that ca be achieved by
		 * using the full node resources divided by the with the number of
		 * nodes running tasks for reading but then divided to the actual number of
		 * tasks to account that if on a node not all the cores are reading
		 * then not the full resources are utilized.
		 */
		VarStats output;
		if (inst instanceof ReblockSPInstruction || inst instanceof CSVReblockSPInstruction || inst instanceof LIBSVMReblockSPInstruction) {
			// Reblock instruction initiate one full stage (hadoop read + mapPartitionsToPair()) and initiates another open one (combineByKey());
			UnarySPInstruction uinst = (UnarySPInstruction) inst;
			VarStats input = getStats((uinst).input1.getName());
			output = getStats((uinst).getOutputVariableName());
			// stage 1: read text file + shuffle write the partitions since then a shuffle is required
			// stage 1, step 1
			double readTime = IOCostUtils.getHadoopReadTime(input, executorMetrics);
			long sizeTextFile = OptimizerUtils.estimateSizeTextOutput(input.getM(), input.getN(), input.getNNZ(), (FileFormat) input.fileInfo[1]);
			RDDStats textRdd = new RDDStats(-1, -1, -1, sizeTextFile, -1);
			// stage 1, step 2
			double shuffleWriteTime = getSparkShuffleWriteTime(textRdd, executorMetrics);
			// stage 2: read partitioned shuffled text object into partitioned binary object
			// stage 2, step 1
			double shuffleReadTime = getSparkShuffleReadTime(textRdd, executorMetrics);
			// init the output rdd characteristics
			output.rddStats.loadCharacteristics();
			output.rddStats.hashPartitioned = true;
			// stage 2, step 2
			double computeTime = SparkCostUtils.getReblockInstTime(uinst.getOpcode(), output.rddStats, executorMetrics);
			// estimate the time as a sum of the first stage the beginning of the next one
			output.rddStats.cost =  readTime + shuffleWriteTime + shuffleReadTime + computeTime;
		} else if (inst instanceof CheckpointSPInstruction) {
			// assume the rdd object is only marked as checkpoint;
			// adding spilling or serializing cost is skipped
			CheckpointSPInstruction cinst = (CheckpointSPInstruction) inst;
			VarStats input = getStats(cinst.input1.getName());
			output = getStats(cinst.input1.getName());
			// only copy the cost from the input rdd
			output.rddStats.cost = input.rddStats.cost;
			// the flag marks that the cost should be reset later
			output.rddStats.checkpoint = true;
		} else if (inst instanceof RandSPInstruction) {
			// Rand instruction takes no RDD input;
			RandSPInstruction rinst = (RandSPInstruction) inst;
			String opcode = rinst.getOpcode();
			if (opcode.equals(SAMPLE_OPCODE)) {
				// sample uses sortByKey() op. and it should be handled differently
				throw new RuntimeException(SAMPLE_OPCODE + " is not supported yet");
			}
			output = getStats(rinst.output.getName());
			output.rddStats.loadCharacteristics();

			int randType = -1; // default for non-random object generation operations
			if (rinst.getMinValue() == 0d && rinst.getMaxValue() == 0d) {
				randType = 0;
			} else if (rinst.getSparsity() == 1.0 && rinst.getMinValue() == rinst.getMaxValue()) {
				randType = 1;
			} else if (opcode.equals(RAND_OPCODE) || opcode.equals(FRAME_OPCODE)) {
				randType = 2;
			}
			// no shuffling required, only computing time
			output.rddStats.cost = getRandInstTime(opcode, output.rddStats, randType, executorMetrics);
		} else if (inst instanceof UnaryMatrixSPInstruction || inst instanceof UnaryFrameSPInstruction) {
			// this instruction adds a map() to an open stage
			UnarySPInstruction uinst = (UnarySPInstruction) inst;
			VarStats input = getStats((uinst).input1.getName());
			// handle input rdd loading
			double loadTime = loadRDDStatsAndEstimateTime(input);
			// init the output rdd characteristics
			output = getStats((uinst).getOutputVariableName());
			output.rddStats.loadCharacteristics();
			output.rddStats.hashPartitioned = input.rddStats.hashPartitioned;
			//compute time only
			output.rddStats.cost = loadTime + SparkCostUtils.getUnaryInstTime(uinst.getOpcode(), input.rddStats, executorMetrics);
		} else if (inst instanceof AggregateUnarySPInstruction || inst instanceof AggregateUnarySketchSPInstruction) {
			UnarySPInstruction auinst = (UnarySPInstruction) inst;
			VarStats input = getStats((auinst).input1.getName());
			// handle input rdd loading
			double loadTime = loadRDDStatsAndEstimateTime(input);
			// init the output rdd characteristics
			output = getStats((auinst).getOutputVariableName());
			output.rddStats.loadCharacteristics();
			// mapping cost represented by the computation time
			double computationTime = SparkCostUtils.getAggUnaryInstTime(inst.getOpcode(), input.rddStats, output.rddStats, executorMetrics);

			AggBinaryOp.SparkAggType aggType = (inst instanceof AggregateUnarySPInstruction)?
					((AggregateUnarySPInstruction) auinst).getAggType():
					((AggregateUnarySketchSPInstruction) auinst).getAggType();
			if (aggType == AggBinaryOp.SparkAggType.SINGLE_BLOCK) {
				// loading RDD to the driver (CP) explicitly (not triggered by CP instruction)
				output.rddStats.isCollected = true;
				// time = computation time (time for transferring is ignored due the small object size)
				output.rddStats.cost = computationTime;
			}
			// none and multi block aggregations preserve the number of partitions;
			output.rddStats.numPartitions = input.rddStats.numPartitions;
			if (aggType == AggBinaryOp.SparkAggType.MULTI_BLOCK){
				// operates over 2 stages: computation time + shuffle time (combineByKey);
				double shuffleWriteTime = IOCostUtils.getSparkShuffleWriteTime(output.rddStats, executorMetrics);
				double shuffleReadTime = IOCostUtils.getSparkShuffleReadTime(output.rddStats, executorMetrics);
				output.rddStats.hashPartitioned = true;
				output.rddStats.cost = computationTime + shuffleWriteTime + shuffleReadTime;
			} else {  // aggType == AggBinaryOp.SparkAggType.NONE
				output.rddStats.hashPartitioned = input.rddStats.hashPartitioned;
				// no reshuffling -> inst is packed with the next spark operation
				output.rddStats.cost = computationTime;
			}
		} else if (inst instanceof BinarySPInstruction) {
			BinarySPInstruction binst = (BinarySPInstruction) inst;
			String opcode = binst.getOpcode();
			VarStats input1 = getStatsWithDefaultScalar((binst).input1.getName());
			// handle input rdd loading
			double loadTime = loadRDDStatsAndEstimateTime(input1);
			VarStats input2 = getStatsWithDefaultScalar((binst).input2.getName());
			// handle input rdd loading
			loadTime += loadRDDStatsAndEstimateTime(input2);
			// init the output rdd characteristics
			output = getStats((binst).getOutputVariableName());
			output.rddStats.loadCharacteristics();

			double executionTime = 0;
			if (inst instanceof BinaryMatrixMatrixSPInstruction) {
				if ((input1.rddStats == null || input1.rddStats.distributedSize < 0)
						|| (input2.rddStats == null || input2.rddStats.distributedSize < 0)) {
					throw new RuntimeException("Input RDD statistics not initialized.");
				}

				if (inst instanceof BinaryMatrixBVectorSPInstruction) {
					// the second matrix is always the broadcast one
					executionTime += IOCostUtils.getBroadcastTime(input2, driverMetrics, executorMetrics);
					// flatMapToPair() or ()mapPartitionsToPair invoked -> no shuffling
					output.rddStats.numPartitions = input1.rddStats.numPartitions;
					output.rddStats.hashPartitioned = input1.rddStats.hashPartitioned;
				} else { // regular BinaryMatrixMatrixSPInstruction
					// join() introduces a new stage
					executionTime += IOCostUtils.getSparkShuffleWriteTime(input1.rddStats, executorMetrics) +
							IOCostUtils.getSparkShuffleWriteTime(input2.rddStats, executorMetrics);
					if (input1.rddStats.hashPartitioned) {
						output.rddStats.numPartitions = input1.rddStats.numPartitions;
						if (!(input1.rddStats.numPartitions == input2.rddStats.numPartitions)) {
							// shuffle the second matrix for join()
							executionTime += IOCostUtils.getSparkShuffleReadTime(input2.rddStats, executorMetrics);
						} // else no shuffle read needed for join()
					} else if (input2.rddStats.hashPartitioned) {
						output.rddStats.numPartitions = input2.rddStats.numPartitions;
						if (!(input1.rddStats.numPartitions == input2.rddStats.numPartitions)) {
							// shuffle the second matrix for join()
							executionTime += IOCostUtils.getSparkShuffleReadTime(input2.rddStats, executorMetrics);
						} // else no shuffle read needed for join()
					} else {
						// repartition needed
						output.rddStats.numPartitions = 2 * output.rddStats.numPartitions;
						executionTime += IOCostUtils.getSparkShuffleReadTime(output.rddStats, executorMetrics);
					}
					output.rddStats.hashPartitioned = true;
				}
			} else if (inst instanceof BinaryMatrixScalarSPInstruction) {
				if ((input1.rddStats == null || input1.rddStats.distributedSize < 0)
						&& (input2.rddStats == null || input2.rddStats.distributedSize < 0)) {
					throw new RuntimeException("Input RDD statistics not initialized.");
				}
				output.rddStats.hashPartitioned = (input1.rddStats != null)? input1.rddStats.hashPartitioned : input2.rddStats.hashPartitioned;
				// only mapValues() invoked -> no shuffling
			} else if (inst instanceof BinaryFrameMatrixSPInstruction || inst instanceof BinaryFrameFrameSPInstruction) {
				throw new RuntimeException("Handling binary instructions for frames not handled yet.");
			} else {
				throw new RuntimeException("Not supported binary instruction: "+inst);
			}

			double computeTime = SparkCostUtils.getBinaryInstTime(opcode, input1.rddStats, input2.rddStats, output.rddStats, executorMetrics);
			output.rddStats.cost = loadTime + executionTime + computeTime;
		} else {
			throw new RuntimeException("Unsupported instruction: " + inst.getOpcode());
		}
		_lineage.add(output);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Helpers for handling stats and estimating time related to their corresponding variables  //
	//////////////////////////////////////////////////////////////////////////////////////////////

	private double loadCPVarStatsAndEstimateTime(VarStats input) throws CostEstimationException {
		if (input.isScalar() || input.allocatedMemory > 0) return 0.0;
		// loading from RDD
		if (input.rddStats != null && input.fileInfo == null) { // input.fileInfo == null false for var output of reblock/checkpoint instructions
			double computeAndLoadTime = getTimeEstimateSparkJob(input);
			if (input.allocatedMemory > 0) {
				// for the cases that an SPInstruction invokes action explicitly
				return computeAndLoadTime;
			}
			if (OptimizerUtils.checkSparkCollectMemoryBudget(input.characteristics, (localMemoryLimit - freeLocalMemory), false)) { // .collect()
				computeAndLoadTime += IOCostUtils.getSparkCollectTime(input.rddStats, driverMetrics, executorMetrics);
			} else { // redirect through HDFS
				// time for writing to HDFS on executors and reading back on driver
				computeAndLoadTime += IOCostUtils.getHadoopWriteTime(input, executorMetrics) +
						IOCostUtils.getDiskReadTime(input, driverMetrics);
			}
			putInMemory(input);
			return computeAndLoadTime;
		} // (input.rddStats != null && input.fileInfo != null) true only for var outputs of reblock and checkpoint instructions possible
		// loading from a file
		if (input.fileInfo == null || input.fileInfo.length != 2) {
			throw new DMLRuntimeException("Time estimation is not possible without file info.");
		}
		else if (!input.fileInfo[0].equals(HDFS_SOURCE_IDENTIFIER) && !input.fileInfo[0].equals(S3_SOURCE_IDENTIFIER)) {
			throw new DMLRuntimeException("Time estimation is not possible for data source: "+ input.fileInfo[0]);
		}
		input.allocatedMemory = OptimizerUtils.estimateSizeExactSparsity(input.characteristics);
		putInMemory(input);
		return IOCostUtils.getDiskReadTime(input, driverMetrics);
	}

	private void putInMemory(VarStats output) throws CostEstimationException {
		if (output.isScalar()) return;
		if (freeLocalMemory - output.allocatedMemory < 0)
			throw new CostEstimationException("Insufficient local memory");
		freeLocalMemory -= output.allocatedMemory;
	}

	private void removeFromMemory(VarStats input) {
		if (input == null) return; // scalars or variables never put in memory
		if (input.varName.equals("beta") || input.varName.equals("_mVar341") || input.varName.equals("_mVar339") || input.varName.equals("newbeta"))
			System.out.println("fuck");
		if (input.allocatedMemory > 0) freeLocalMemory += input.allocatedMemory;
		if (input.rddStats != null) {
			freeDistributedMemory += input.rddStats.distributedSize;
			input.rddStats = null;
		}
		input.allocatedMemory = -1;
	}

	/**
	 * This method serves a mein rule at the mechanism for
	 * estimation the execution time of Spark instructions:
	 * it estimates the time for distributing existing CP variable
	 * or sets the estimated time as time needed for computing the
	 * input variable on Spark.
	 * @param input input statistics
	 * @return time (seconds) for loading the corresponding variable
	 */
	private double loadRDDStatsAndEstimateTime(VarStats input) {
		if (input.isScalar()) return 0.0;

		double ret;
		if (input.rddStats == null) { // rdd is to be distributed by the CP
			input.rddStats = new RDDStats(input);
			RDDStats inputRDD = input.rddStats;
			// load in advance to estimate the distributed size of the RDD object
			inputRDD.loadCharacteristics();
			if (input.allocatedMemory >= 0) { // generated object locally
				if (inputRDD.distributedSize < freeLocalMemory || !_parRDDs.reserve(inputRDD.distributedSize)) {
					// TODO: consider if isDirty flag can be useful in the cost estimation to check for need ot write to HDFS
					// in this case transfer the data object over HDF (first set the fileInfo of the input)
					input.fileInfo = new Object[] {HDFS_SOURCE_IDENTIFIER, FileFormat.BINARY};
					ret = IOCostUtils.getDiskWriteTime(input, driverMetrics);
					ret += IOCostUtils.getHadoopReadTime(input, executorMetrics);
				} else {
					ret = IOCostUtils.getParallelizeTime(inputRDD, driverMetrics, executorMetrics);
				}
			} else { // on hdfs
				if (input.fileInfo == null || input.fileInfo.length != 2)
					throw new DMLRuntimeException("File info missing for a file to be read on Spark.");
				ret = IOCostUtils.getHadoopReadTime(input, executorMetrics);
			}
		} else if (input.rddStats.distributedSize > 0) {
			// if input RDD size is initiated -> cost should be calculated
			// transfer the cost to the output rdd for lineage proper handling
			ret = input.rddStats.cost;
			if (input.rddStats.checkpoint) {
				// cost of checkpoint var transferred only once
				input.rddStats.cost = 0;
			}
			if (input.rddStats.isCollected) {
				// entering unhandled case
				throw new RuntimeException("Using collected RDDs is not handled");
			}
		} else {
			throw new RuntimeException("Initialized RDD stats without initialized data characteristics is undefined behaviour");
		}
		return ret;
	}
}
