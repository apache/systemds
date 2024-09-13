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
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.MapMult;
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
	private static final int DEFAULT_NUM_ITER = 15;

	// Non-static members
	private final Program _program;
	private final IOCostUtils.IOMetrics driverMetrics;
	private final IOCostUtils.IOMetrics executorMetrics;
	private final SparkExecutionContext.MemoryManagerParRDDs _parRDDs;
	// declare here the hashmaps
	private final HashMap<String, VarStats> _stats;
	private final HashSet<String> _functions;
	private final long localMemoryLimit; // refers to the drivers JVM memory
	private long freeLocalMemory;

	/**
	 * Entry point for estimating the execution time of a program.
	 * @param program compiled runtime program
	 * @return estimated time for execution of the program
	 * given the resources set in {@link SparkExecutionContext}
	 * @throws CostEstimationException in case of errors
	 */
	public static double estimateExecutionTime(Program program, CloudInstance driverNode, CloudInstance executorNode) throws CostEstimationException {
		CostEstimator estimator = new CostEstimator(program, driverNode, executorNode);
		return estimator.getTimeEstimate();
	}
	public CostEstimator(Program program, CloudInstance driverNode, CloudInstance executorNode) {
		_program = program;
		driverMetrics = new IOCostUtils.IOMetrics(driverNode);
		executorMetrics = executorNode != null? new IOCostUtils.IOMetrics(executorNode) : null;
		// initialize here the hashmaps
		_stats = new HashMap<>();
		_functions = new HashSet<>();
		localMemoryLimit = (long) OptimizerUtils.getLocalMemBudget();
		freeLocalMemory = localMemoryLimit;
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
					VarStats outputStats = getStats(varName);
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
		}
	}

	public double getTimeEstimateInst(Instruction inst) throws CostEstimationException {
		double timeEstimate;
		if (inst instanceof CPInstruction) {
			timeEstimate = getTimeEstimateCPInst((CPInstruction)inst);
		} else { // inst instanceof SPInstruction
			timeEstimate = parseSPInst((SPInstruction) inst);
		}
		return timeEstimate;
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

	/**
	 * Parse a Spark instruction, and it stores the corresponding
	 * cost for computing the output variable in the RDD statistics'
	 * object related to that variable.
	 * This method is responsible for initializing the corresponding
	 * {@code RDDStats} object for each output variable, including for
	 * outputs that are explicitly brought back to CP (Spark action within the instruction).
	 * It returns the time estimate only for those instructions that bring the
	 * output explicitly to CP. For the rest, the estimated time (cost) is
	 * stored as part of the corresponding RDD statistics, emulating the
	 * lazy evaluation execution of Spark.
	 *
	 * @param inst Spark instruction for parsing
	 * @return if explicit action, estimated time in seconds, else always 0
	 */
	public double parseSPInst(SPInstruction inst) throws CostEstimationException {
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
			UnarySPInstruction uinst = (UnarySPInstruction) inst;
			VarStats input = getStats((uinst).input1.getName());
			output = getStats((uinst).getOutputVariableName());
			output.rddStats = new RDDStats(output);
			output.fileInfo = input.fileInfo;
			// the resulting binary rdd is being hash-partitioned after the reblock
			output.rddStats.hashPartitioned = true;
			output.rddStats.cost =  SparkCostUtils.getReblockInstTime(inst.getOpcode(), input, output, executorMetrics);
		} else if (inst instanceof CheckpointSPInstruction) {
			CheckpointSPInstruction cinst = (CheckpointSPInstruction) inst;
			VarStats input = getStats(cinst.input1.getName());
			output = getStats(cinst.input1.getName());
			output.rddStats = new RDDStats(output);

			output.fileInfo = input.fileInfo;
			output.rddStats.checkpoint = true;
			// assume the rdd object is only marked as checkpoint;
			// adding spilling or serializing cost is skipped
			output.rddStats.cost = input.rddStats.cost;
		} else if (inst instanceof RandSPInstruction) {
			// Rand instruction takes no RDD input;
			RandSPInstruction rinst = (RandSPInstruction) inst;
			String opcode = rinst.getOpcode();
			int randType = -1; // default for non-random object generation operations
			if (opcode.equals(RAND_OPCODE) || opcode.equals(FRAME_OPCODE)) {
				if (rinst.getMinValue() == 0d && rinst.getMaxValue() == 0d) { // empty matrix
					randType = 0;
				} else if (rinst.getSparsity() == 1.0 && rinst.getMinValue() == rinst.getMaxValue()) { // allocate, array fill
					randType = 1;
				} else { // full rand
 					randType = 2;
				}
			}
			output = getStats(rinst.output.getName());
			output.rddStats = new RDDStats(output);
			output.rddStats.cost = getRandInstTime(opcode, randType, output, executorMetrics);
		} else if (inst instanceof AggregateUnarySPInstruction || inst instanceof AggregateUnarySketchSPInstruction) {
			UnarySPInstruction auinst = (UnarySPInstruction) inst;
			VarStats input = getStats((auinst).input1.getName());
			double loadTime = loadRDDStatsAndEstimateTime(input);

			output = getStats((auinst).getOutputVariableName());
			output.rddStats = new RDDStats(output);
			output.rddStats.cost = loadTime + SparkCostUtils.getAggUnaryInstTime(auinst, input, output, executorMetrics);
		} else if (inst instanceof IndexingSPInstruction) {
			IndexingSPInstruction ixdinst = (IndexingSPInstruction) inst;
			boolean isLeftCacheType = (inst instanceof MatrixIndexingSPInstruction &&
					((MatrixIndexingSPInstruction) ixdinst).getLixType() == LeftIndex.LixCacheType.LEFT);
			VarStats input1; // always assigned
			VarStats input2 = null; // assigned only if case of indexing
			double loadTime = 0;
			if (ixdinst.getOpcode().toLowerCase().contains("left")) {
				if (isLeftCacheType) {
					input1 = getStats(ixdinst.input2.getName());
					input2 = getStats(ixdinst.input1.getName());
				} else {
					input1 = getStats(ixdinst.input1.getName());
					input2 = getStats(ixdinst.input2.getName());
				}

				if (ixdinst.getOpcode().equals(LeftIndex.OPCODE)) {
					loadTime += loadRDDStatsAndEstimateTime(input2);
				} else { // mapLeftIndex
					loadTime += loadCPVarStatsAndEstimateTime(input2);
				}
			} else {
				input1 = getStats(ixdinst.input1.getName());
			}
			loadTime += loadRDDStatsAndEstimateTime(input1);

//			VarStats rowLower = getStatsWithDefaultScalar(ixdinst.getRowLower().getName());
//			VarStats rowUpper = getStatsWithDefaultScalar(ixdinst.getRowUpper().getName());
//			VarStats colLower = getStatsWithDefaultScalar(ixdinst.getColLower().getName());
//			VarStats colUpper = getStatsWithDefaultScalar(ixdinst.getColUpper().getName());
			output = getStats(ixdinst.getOutputVariableName());
			output.rddStats = new RDDStats(output);
			output.rddStats.cost = loadTime +
					SparkCostUtils.getIndexingInstTime(ixdinst, input1, input2, output, driverMetrics, executorMetrics);
		} else if (inst instanceof UnarySPInstruction) { // general unary handling body; put always after all the rest blocks for unary
			UnarySPInstruction uinst = (UnarySPInstruction) inst;
			VarStats input = getStats((uinst).input1.getName());
			double loadTime = loadRDDStatsAndEstimateTime(input);
			output = getStats((uinst).getOutputVariableName());
			output.rddStats = new RDDStats(output);
			if (uinst instanceof UnaryMatrixSPInstruction || inst instanceof UnaryFrameSPInstruction) {
				output.rddStats.cost = loadTime + SparkCostUtils.getUnaryInstTime(uinst.getOpcode(), input, output, executorMetrics);
			} else if (uinst instanceof ReorgSPInstruction || inst instanceof MatrixReshapeSPInstruction) {
				output.rddStats.cost = loadTime + SparkCostUtils.getReorgInstTime(uinst, input, output, executorMetrics);
			} else if (uinst instanceof TsmmSPInstruction || inst instanceof Tsmm2SPInstruction) {
				output.rddStats.cost = loadTime + SparkCostUtils.getTSMMInstTime(uinst, input, output, driverMetrics, executorMetrics);
			} else if (uinst instanceof CentralMomentSPInstruction) {
				VarStats weights = null;
				if (uinst.input3 != null) {
					weights = getStats(uinst.input2.getName());
					loadTime += loadRDDStatsAndEstimateTime(weights);
				}
				output.rddStats.cost = loadTime +
						SparkCostUtils.getCentralMomentInstTime((CentralMomentSPInstruction) uinst, input, weights, output, executorMetrics);
			} else if (inst instanceof CastSPInstruction) {
				output.rddStats.cost = loadTime + SparkCostUtils.getCastInstTime((CastSPInstruction) inst, input, output, executorMetrics);
			} else if (inst instanceof QuantileSortSPInstruction) {
				VarStats weights = null;
				if (uinst.input2 != null) {
					weights = getStats(uinst.input2.getName());
					loadTime += loadRDDStatsAndEstimateTime(weights);
				}
				output.rddStats.cost = loadTime +
						SparkCostUtils.getQSortInstTime((QuantileSortSPInstruction) uinst, input, weights, output, executorMetrics);
			} else {
				throw new RuntimeException("Unsupported Unary Spark instruction of type " + inst.getClass().getName());
			}
		} else if (inst instanceof BinaryFrameFrameSPInstruction || inst instanceof BinaryFrameMatrixSPInstruction || inst instanceof BinaryMatrixMatrixSPInstruction || inst instanceof BinaryMatrixScalarSPInstruction) {
			BinarySPInstruction binst = (BinarySPInstruction) inst;
			VarStats input1 = getStatsWithDefaultScalar((binst).input1.getName());
			VarStats input2 = getStatsWithDefaultScalar((binst).input2.getName());
			// handle input rdd loading
			double loadTime = loadRDDStatsAndEstimateTime(input1);
			if (inst instanceof  BinaryMatrixBVectorSPInstruction) {
				loadTime += loadCPVarStatsAndEstimateTime(input2);
			} else {
				loadTime += loadRDDStatsAndEstimateTime(input2);
			}

			output = getStats((binst).getOutputVariableName());
			output.rddStats = new RDDStats(output);

			output.rddStats.cost = loadTime +
					SparkCostUtils.getBinaryInstTime(inst, input1, input2, output, driverMetrics, executorMetrics);
		} else if (inst instanceof AppendSPInstruction) {
			AppendSPInstruction ainst = (AppendSPInstruction) inst;
			VarStats input1 = getStats(ainst.input1.getName());
			double loadTime = loadRDDStatsAndEstimateTime(input1);
			VarStats input2 = getStats(ainst.input2.getName());
			if (ainst instanceof AppendMSPInstruction) {
				loadTime += loadCPVarStatsAndEstimateTime(input2);
			} else {
				loadTime += loadRDDStatsAndEstimateTime(input2);
			}
			output = getStats(ainst.getOutputVariableName());
			output.rddStats = new RDDStats(output);

			output.rddStats.cost = loadTime + SparkCostUtils.getAppendInstTime(ainst, input1, input2, output, driverMetrics, executorMetrics);
		} else if (inst instanceof AggregateBinarySPInstruction || inst instanceof PmmSPInstruction || inst instanceof PMapmmSPInstruction || inst instanceof ZipmmSPInstruction) {
			BinarySPInstruction binst = (BinarySPInstruction) inst;
			VarStats input1, input2;
			double loadTime = 0;
			if (binst instanceof MapmmSPInstruction || binst instanceof PmmSPInstruction) {
				MapMult.CacheType cacheType = binst instanceof MapmmSPInstruction?
						((MapmmSPInstruction) binst).getCacheType() :
						((PmmSPInstruction) binst).getCacheType();
				if (cacheType.isRight()) {
					input1 = getStats(binst.input1.getName());
					input2 = getStats(binst.input2.getName());
				} else {
					input1 = getStats(binst.input2.getName());
					input2 = getStats(binst.input1.getName());
				}
				loadTime += loadRDDStatsAndEstimateTime(input1);
				loadTime += loadCPVarStatsAndEstimateTime(input2);
			} else {
				input1 = getStats(binst.input1.getName());
				input2 = getStats(binst.input2.getName());
				loadTime += loadRDDStatsAndEstimateTime(input1);
				loadTime += loadRDDStatsAndEstimateTime(input2);
			}
			output = getStats(binst.getOutputVariableName());
			output.rddStats = new RDDStats(output);
			output.rddStats.cost = loadTime +
					SparkCostUtils.getMatMulInstTime(binst, input1, input2, output, driverMetrics, executorMetrics);
		} else if (inst instanceof MapmmChainSPInstruction) {
			MapmmChainSPInstruction mmchaininst = (MapmmChainSPInstruction) inst;
			VarStats input1 = getStats(mmchaininst.input1.getName());
			VarStats input2 = getStats(mmchaininst.input1.getName());
			VarStats input3 = null;
			double loadTime = loadRDDStatsAndEstimateTime(input1) + loadCPVarStatsAndEstimateTime(input2);
			if (mmchaininst.input3 != null) {
				input3 = getStats(mmchaininst.input3.getName());
				loadTime += loadCPVarStatsAndEstimateTime(input3);
			}
			output = getStats(mmchaininst.output.getName());
			output.rddStats = new RDDStats(output);
			output.rddStats.cost = loadTime +
					SparkCostUtils.getMatMulChainInstTime(mmchaininst, input1, input2, input3, output, driverMetrics, executorMetrics);
		} else if (inst instanceof WriteSPInstruction) {

			// return time estimate here since no corresponding RDD statistics exist
			return 0;
		}
//		else if (inst instanceof CumulativeOffsetSPInstruction) {
//
//		} else if (inst instanceof CovarianceSPInstruction) {
//
//		} else if (inst instanceof QuantilePickSPInstruction) {
//
//		} else if (inst instanceof TernarySPInstruction) {
//
//		} else if (inst instanceof AggregateTernarySPInstruction) {
//
//		} else if (inst instanceof QuaternarySPInstruction) {
//
//		} else if (inst instanceof CtableSPInstruction) {
//
//		}
		else {
			throw new RuntimeException("Unsupported instruction: " + inst.getOpcode());
		}
		// output.rdd should be always initialized at this point
		if (output.rddStats.isCollected) {
			double ret = output.rddStats.cost;
			output.rddStats = null;
			return ret;
		}
		return 0;
	}

	public double getTimeEstimateSparkJob(VarStats varToCollect) {
		if (varToCollect.rddStats == null) {
			throw new RuntimeException("Missing RDD statistics for estimating execution time for Spark Job");
		}
		double computeTime = varToCollect.rddStats.cost;
		double collectTime;
		if (OptimizerUtils.checkSparkCollectMemoryBudget(varToCollect.characteristics, freeLocalMemory, false)) {
			// use Spark collect()
			collectTime = IOCostUtils.getSparkCollectTime(varToCollect.rddStats, driverMetrics, executorMetrics);
		} else {
			// redirect through HDFS (writing to HDFS on executors and reading back on driver)
			collectTime = IOCostUtils.getHadoopWriteTime(varToCollect, executorMetrics) +
					IOCostUtils.getFileSystemReadTime(varToCollect, driverMetrics);
		}
		if (varToCollect.rddStats.checkpoint) {
			varToCollect.rddStats.cost = 0;
		} else {
			varToCollect.rddStats = null;
		}

		if (computeTime < 0 || collectTime < 0) {
			// detection for functionality bugs
			throw new RuntimeException("Unexpected negative value at estimating Spark Job execution time");
		}
		return computeTime + computeTime;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////
	// Helpers for handling stats and estimating time related to their corresponding variables  //
	//////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * This method emulates the SystemDS mechanism of loading objects into
	 * the CP memory from a file or an existing RDD object.
	 *
	 * @param input variable for loading in CP memory
	 * @return estimated time in seconds for loading into memory
	 */
	private double loadCPVarStatsAndEstimateTime(VarStats input) throws CostEstimationException {
		if (input.isScalar() || input.allocatedMemory > 0) return 0.0;

		double loadTime;
		if (input.rddStats != null && input.fileInfo == null) { // input.fileInfo == null false for var output of reblock/checkpoint instructions
			// loading from RDD
			loadTime = getTimeEstimateSparkJob(input);
		} else {
			// loading from a file
			if (input.fileInfo == null || input.fileInfo.length != 2) {
				throw new DMLRuntimeException("Time estimation is not possible without file info.");
			} else if (!input.fileInfo[0].equals(HDFS_SOURCE_IDENTIFIER) && !input.fileInfo[0].equals(S3_SOURCE_IDENTIFIER)) {
				throw new DMLRuntimeException("Time estimation is not possible for data source: " + input.fileInfo[0]);
			}
			loadTime = IOCostUtils.getFileSystemReadTime(input, driverMetrics);
		}
		input.allocatedMemory = OptimizerUtils.estimateSizeExactSparsity(input.characteristics);
		putInMemory(input);
		return loadTime;
	}

	private void putInMemory(VarStats output) throws CostEstimationException {
		if (output.isScalar()) return;
		if (freeLocalMemory - output.allocatedMemory < 0)
			throw new CostEstimationException("Insufficient local memory");
		freeLocalMemory -= output.allocatedMemory;
	}

	private void removeFromMemory(VarStats input) {
		if (input == null) return; // scalars or variables never put in memory
		if (input.allocatedMemory > 0) {
			freeLocalMemory += input.allocatedMemory;
			if (freeLocalMemory > localMemoryLimit) {
				// detection of functionality bugs
				throw new RuntimeException("Unexpectedly large amount of freed CP memory");
			}
		}
		if (input.rddStats != null) {
			input.rddStats = null;
		}
		input.allocatedMemory = -1;
	}

	/**
	 * This method serves a main rule at the mechanism for
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
			if (input.allocatedMemory >= 0) { // generated object locally
				if (inputRDD.distributedSize < freeLocalMemory || !_parRDDs.reserve(inputRDD.distributedSize)) {
					// TODO: consider if isDirty flag can be useful in the cost estimation to check for need ot write to HDFS
					// in this case transfer the data object over HDF (first set the fileInfo of the input)
					input.fileInfo = new Object[] {HDFS_SOURCE_IDENTIFIER, FileFormat.BINARY};
					ret = IOCostUtils.getDiskWriteTime(input, driverMetrics);
					ret += IOCostUtils.getHadoopReadTime(input, executorMetrics);
				} else {
					ret = IOCostUtils.getSparkParallelizeTime(inputRDD, driverMetrics, executorMetrics);
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
		} else {
			throw new RuntimeException("Initialized RDD stats without initialized data characteristics is undefined behaviour");
		}
		return ret;
	}
}
