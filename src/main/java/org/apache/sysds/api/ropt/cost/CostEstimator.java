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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.*;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData.CacheStatus;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.instructions.spark.SPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnarySPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.runtime.meta.TensorCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.*;

public abstract class CostEstimator
{
	protected static final Log LOG = LogFactory.getLog(CostEstimator.class.getName());
	
	private static final int DEFAULT_NUMITER = 15;

	private static final long[] DEFAULT_TENSOR_DIMS = new long[]{10, 10, 10};

	protected static final VarStats _unknownStats = new VarStats(new MatrixCharacteristics(-1,-1,-1,-1),false, Types.DataType.UNKNOWN);
	protected static final VarStats _scalarStats = new VarStats(new MatrixCharacteristics(1,1,1,1),true, Types.DataType.SCALAR);
	
	public double getTimeEstimate(Program rtprog, LocalVariableMap vars, HashMap<String,VarStats> stats) {
		double costs = 0;

		maintainVariableStatistics(vars, stats);
						
		//get cost estimate
		for( ProgramBlock pb : rtprog.getProgramBlocks() )
			costs += rGetTimeEstimate(pb, stats, new HashSet<String>(), true);
		
		return costs;
	}

	private double rGetTimeEstimate(ProgramBlock pb, HashMap<String, VarStats> stats, HashSet<String> memoFunc, boolean recursive) {
		double ret = 0;
		
		if (pb instanceof WhileProgramBlock) {
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			if( recursive )
				for (ProgramBlock pb2 : tmp.getChildBlocks())
					ret += rGetTimeEstimate(pb2, stats, memoFunc, recursive);
			ret *= DEFAULT_NUMITER;
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock tmp = (IfProgramBlock)pb;
			if( recursive ) {
				for( ProgramBlock pb2 : tmp.getChildBlocksIfBody() )
					ret += rGetTimeEstimate(pb2, stats, memoFunc, recursive);
				if( tmp.getChildBlocksElseBody()!=null )
					for( ProgramBlock pb2 : tmp.getChildBlocksElseBody() ){
						ret += rGetTimeEstimate(pb2, stats, memoFunc, recursive);
						ret /= 2; //weighted sum
					}
			}
		}
		else if (pb instanceof ForProgramBlock) { //includes ParFORProgramBlock
			ForProgramBlock tmp = (ForProgramBlock)pb;
			if( recursive )
				for( ProgramBlock pb2 : tmp.getChildBlocks() )
					ret += rGetTimeEstimate(pb2, stats, memoFunc, recursive);
			
			ret *= getNumIterations(tmp);
		}
		else if ( pb instanceof FunctionProgramBlock ) {
			FunctionProgramBlock tmp = (FunctionProgramBlock) pb;
			if( recursive )
				for( ProgramBlock pb2 : tmp.getChildBlocks() )
					ret += rGetTimeEstimate(pb2, stats, memoFunc, recursive);
		}
		else if( pb instanceof BasicProgramBlock ) 
		{
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			ArrayList<Instruction> tmp = bpb.getInstructions();
			
			for( Instruction inst : tmp )
			{
				if( inst instanceof CPInstruction ) //CP
				{
					//obtain stats from createvar, cpvar, rmvar, rand
					maintainCPInstVariableStatistics((CPInstruction)inst, stats);

					//extract statistics (instruction-specific)
//					Object[] o = extractCPInstStatistics(inst, stats);
//					VarStats[] vs = (VarStats[]) o[0];
//					String[] attr = (String[]) o[1];
					
					//if(LOG.isDebugEnabled())
					//	LOG.debug(inst);
					
					//call time estimation for inst
					ret += getCPInstTimeEstimate((CPInstruction) inst, stats);
					
					if( inst instanceof FunctionCallCPInstruction ) //functions
					{
						FunctionCallCPInstruction finst = (FunctionCallCPInstruction)inst;
						String fkey = DMLProgram.constructFunctionKey(finst.getNamespace(), finst.getFunctionName());
						//awareness of recursive functions, missing program
						if( !memoFunc.contains(fkey) && pb.getProgram()!=null ) 
						{
							if(LOG.isDebugEnabled())
								LOG.debug("Begin Function "+fkey);
							
							memoFunc.add(fkey);
							Program prog = pb.getProgram();
							FunctionProgramBlock fpb = prog.getFunctionProgramBlock(
								finst.getNamespace(), finst.getFunctionName());
							ret += rGetTimeEstimate(fpb, stats, memoFunc, recursive);
							memoFunc.remove(fkey);
							
							if(LOG.isDebugEnabled())
								LOG.debug("End Function "+fkey);
						}
					}
				} else if ( inst instanceof SPInstruction) //Spark
				{
					//obtain stats from createvar, cpvar, rmvar, rand
					maintainSPInstVariableStatistics((SPInstruction)inst, stats);

					//extract statistics (instruction-specific)
					Object[] o = extractSPInstStatistics(inst, stats);
					VarStats[] vs = null;
					String[] attr = null;

					//if(LOG.isDebugEnabled())
					//	LOG.debug(inst);

					//call time estimation for inst
					// TODO: add ret += getSPInstTimeEstimate(inst, vs, attr); when extractSPInstStatistics(inst, stats); is implemented
				}
			}
		}
		
		return ret;
	}

	private Object[] extractSPInstStatistics(Instruction inst, HashMap<String, VarStats> stats) {
		// TODO: implement considering the old extractMRInstStatistics
		return null;
	}

	private void maintainSPInstVariableStatistics(SPInstruction inst, HashMap<String, VarStats> stats) {

		int index = -1;
		for( String varname : inVars )
		{
			VarStats vs = stats.get(varname);
			if( vs==null )
				vs = _unknownStats;
			stats.put(String.valueOf(++index), vs);
		}

		if (inst instanceof UnarySPInstruction) {
			UnarySPInstruction unarySPInstruction = (UnarySPInstruction) inst;
			funarySPInstruction.
		}
	}

	private static void maintainCPInstVariableStatistics(CPInstruction inst, HashMap<String, VarStats> stats)
	{
		if( inst instanceof VariableCPInstruction )
		{
			String optype = inst.getOpcode();
			VariableCPInstruction vinst = (VariableCPInstruction) inst;
			if( optype.equals("createvar") ) {
				DataCharacteristics dataCharacteristics = vinst.getMetaData().getDataCharacteristics();
				Types.DataType dt = vinst.getOutput().getDataType();
				VarStats varStats = new VarStats(dataCharacteristics, false, dt);
				stats.put(vinst.getOutputVariableName(), varStats);
			}
			else if ( optype.equals("cpvar") ) {
				VarStats copiedStats = stats.get(vinst.getInput1().getName());
				stats.put(vinst.getInput2().getName(), copiedStats);
			}
			else if ( optype.equals("mvvar") ) {
				VarStats statsToMove = stats.get(vinst.getInput1().getName());
				stats.remove(vinst.getInput1().getName());
				stats.put(vinst.getInput2().getName(), statsToMove);
			}
			else if( optype.equals("rmvar") ) {
				stats.remove(vinst.getInput1().getName());
			}
		}
		else if( inst instanceof DataGenCPInstruction ){
			DataGenCPInstruction randInst = (DataGenCPInstruction) inst;
			String varname = randInst.getOutputVariableName();

			DataCharacteristics dataCharacteristics;
			if (randInst.getOutput().isTensor()) {
				long[] tensorDims = getTensorDims(randInst);
				double nnz = randInst.getSparsity();
				int i = 0;
				for (long d: tensorDims) {
					nnz *= d;

				}
				dataCharacteristics = new TensorCharacteristics(tensorDims, randInst.getBlocksize(), (long) nnz);
			} else {
				long rlen = randInst.getRows();
				long clen = randInst.getCols();
				int blen = randInst.getBlocksize();
				long nnz = (long) (randInst.getSparsity() * rlen * clen);
				dataCharacteristics = new MatrixCharacteristics(rlen, clen, blen, nnz);

			}
			stats.put(varname, new VarStats(dataCharacteristics, false, Types.DataType.MATRIX));
		}
		else if( inst instanceof StringInitCPInstruction ){
			StringInitCPInstruction iinst = (StringInitCPInstruction) inst;
			String varname = iinst.getOutputVariableName();
			long rlen = iinst.getRows();
			long clen = iinst.getCols();

			MatrixCharacteristics mc = new MatrixCharacteristics(rlen, clen, ConfigurationManager.getBlocksize(), rlen*clen);
			VarStats initStats = new VarStats(mc, false, Types.DataType.MATRIX);
			stats.put(varname, initStats);
		}
		else if( inst instanceof FunctionCallCPInstruction )
		{
			FunctionCallCPInstruction finst = (FunctionCallCPInstruction) inst;
			for( String varname : finst.getBoundOutputParamNames() )
				stats.put(varname, _unknownStats);
		}
	}

	private static long[] getTensorDims(DataGenCPInstruction dataGenInst) {
		long[] tDims;
		CPOperand dims = dataGenInst.getDimsOperand();
		switch (dims.getDataType()) {
			case SCALAR: {
				// Dimensions given as string
				if (dims.getValueType() != Types.ValueType.STRING) {
					throw new DMLRuntimeException("Dimensions have to be passed as list, string, matrix or tensor.");
				}
				if (dims.isLiteral()) {
					ScalarObject s = dims.getLiteral();

					StringTokenizer dimensions = new StringTokenizer(s.getStringValue(), " ");
					tDims = new long[dimensions.countTokens()];
					Arrays.setAll(tDims, (i) -> Integer.parseInt(dimensions.nextToken()));
				} else {
					tDims = DEFAULT_TENSOR_DIMS;
				}

			}
			break;
			case LIST:
			case MATRIX:
			case TENSOR:
				tDims = DEFAULT_TENSOR_DIMS;
				break;
			default:
				throw new DMLRuntimeException("Dimensions have to be passed as list, string, matrix or tensor.");
			}
		return tDims;
	}



	private static void setUnknownStats(VarStats[] vs) {
		vs[0] = _unknownStats;
		vs[1] = _unknownStats;
		vs[2] = _unknownStats;
	}
		
	private static long getNumIterations(ForProgramBlock pb) {
		return OptimizerUtils.getNumIterations(pb, DEFAULT_NUMITER);
	}

	protected abstract double getCPInstTimeEstimate(CPInstruction cpInstruction, HashMap<String, VarStats> stats);

	protected abstract double getSPInstTimeEstimate(Instruction inst, VarStats[] vs, String[] args );


	// --------------- Potentially needed in the future programs --------------------
	private static void maintainVariableStatistics( LocalVariableMap vars, HashMap<String, VarStats> stats ) {
		for( String varname : vars.keySet() ) {
			Data dat = vars.get(varname);
			VarStats vs;
			// TODO: more compact implementation
			if (dat instanceof MatrixObject) //matrix
			{
				MatrixObject mo = (MatrixObject) dat;
				DataCharacteristics dc = mo.getDataCharacteristics();
				boolean inmem = mo.getStatus() == CacheStatus.CACHED;
				vs = new VarStats(dc, inmem, Types.DataType.MATRIX);
			} else if (dat instanceof FrameObject) //frame
			{
				FrameObject fo = (FrameObject) dat;
				DataCharacteristics dc = fo.getDataCharacteristics();
				boolean inmem = fo.getStatus() == CacheStatus.CACHED;
				vs = new VarStats(dc, inmem, Types.DataType.FRAME);
			} else if (dat instanceof TensorObject) // tensor
			{
				TensorObject to = (TensorObject) dat;
				DataCharacteristics dc = to.getDataCharacteristics();
				boolean inmem = to.getStatus()==CacheStatus.CACHED;
				vs = new VarStats(dc, inmem, Types.DataType.TENSOR);
			}
			else //scalar note: and potentially ListObject etc.
			{
				vs = _scalarStats;
			}

			stats.put(varname, vs);
		}
	}

	protected String replaceInstructionPatch( String inst )
	{
		String ret = inst;
		while( ret.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ) {
			int index1 = ret.indexOf(Lop.VARIABLE_NAME_PLACEHOLDER);
			int index2 = ret.indexOf(Lop.VARIABLE_NAME_PLACEHOLDER, index1+1);
			String replace = ret.substring(index1,index2+1);
			ret = ret.replaceAll(replace, "1");
		}

		return ret;
	}

	// old

	private static Object[] extractCPInstStatistics( Instruction inst, HashMap<String, VarStats> stats)
	{
		Object[] ret = new Object[2]; //stats, attrs
		VarStats[] vs = new VarStats[3];
		String[] attr = null;

		if( inst instanceof UnaryCPInstruction ) {
			if( inst instanceof DataGenCPInstruction ) {
				DataGenCPInstruction rinst = (DataGenCPInstruction) inst;
				vs[0] = _unknownStats;
				vs[1] = _unknownStats;
				vs[2] = stats.get( rinst.output.getName() );

				//prepare attributes for cost estimation
				int type = 2; //full rand
				if( rinst.getMinValue() == 0.0 && rinst.getMaxValue() == 0.0 )
					type = 0;
				else if( rinst.getSparsity() == 1.0 && rinst.getMinValue() == rinst.getMaxValue() )
					type = 1;
				attr = new String[]{String.valueOf(type)};
			}
			else if( inst instanceof StringInitCPInstruction ) {
				StringInitCPInstruction rinst = (StringInitCPInstruction) inst;
				vs[0] = _unknownStats;
				vs[1] = _unknownStats;
				vs[2] = stats.get( rinst.output.getName() );
			}
			else { //general unary
				UnaryCPInstruction uinst = (UnaryCPInstruction) inst;
				vs[0] = stats.get( uinst.input1.getName() );
				vs[1] = _unknownStats;
				vs[2] = stats.get( uinst.output.getName() );

				if( vs[0] == null ) //scalar input, e.g., print
					vs[0] = _scalarStats;
				if( vs[2] == null ) //scalar output
					vs[2] = _scalarStats;

				if( inst instanceof MMTSJCPInstruction ) {
					String type = ((MMTSJCPInstruction)inst).getMMTSJType().toString();
					attr = new String[]{type};
				}
				else if( inst instanceof AggregateUnaryCPInstruction ) {
					String[] parts = InstructionUtils.getInstructionParts(inst.toString());
					String opcode = parts[0];
					if( opcode.equals("cm") )
						attr = new String[]{parts[parts.length-2]};
				}
			}
		}
		else if( inst instanceof BinaryCPInstruction ) {
			BinaryCPInstruction binst = (BinaryCPInstruction) inst;
			vs[0] = stats.get( binst.input1.getName() );
			vs[1] = stats.get( binst.input2.getName() );
			vs[2] = stats.get( binst.output.getName() );

			if( vs[0] == null ) //scalar input,
				vs[0] = _scalarStats;
			if( vs[1] == null ) //scalar input,
				vs[1] = _scalarStats;
			if( vs[2] == null ) //scalar output
				vs[2] = _scalarStats;
		}
		else if( inst instanceof AggregateTernaryCPInstruction ) {
			AggregateTernaryCPInstruction binst = (AggregateTernaryCPInstruction) inst;
			//of same dimension anyway but missing third input
			vs[0] = stats.get( binst.input1.getName() );
			vs[1] = stats.get( binst.input2.getName() );
			vs[2] = stats.get( binst.output.getName() );

			if( vs[0] == null ) //scalar input,
				vs[0] = _scalarStats;
			if( vs[1] == null ) //scalar input,
				vs[1] = _scalarStats;
			if( vs[2] == null ) //scalar output
				vs[2] = _scalarStats;
		}
		else if( inst instanceof ParameterizedBuiltinCPInstruction ) {
			//ParameterizedBuiltinCPInstruction pinst = (ParameterizedBuiltinCPInstruction) inst;
			String[] parts = InstructionUtils.getInstructionParts(inst.toString());
			String opcode = parts[0];
			if( opcode.equals("groupedagg") ) {
				HashMap<String,String> paramsMap = ParameterizedBuiltinCPInstruction.constructParameterMap(parts);
				String fn = paramsMap.get("fn");
				String order = paramsMap.get("order");
				AggregateOperationTypes type = CMOperator.getAggOpType(fn, order);
				attr = new String[]{String.valueOf(type.ordinal())};
			}
			else if( opcode.equals("rmempty") ) {
				HashMap<String,String> paramsMap = ParameterizedBuiltinCPInstruction.constructParameterMap(parts);
				attr = new String[]{String.valueOf(paramsMap.get("margin").equals("rows")?0:1)};
			}

			vs[0] = stats.get( parts[1].substring(7).replaceAll(Lop.VARIABLE_NAME_PLACEHOLDER, "") );
			vs[1] = _unknownStats; //TODO
			vs[2] = stats.get( parts[parts.length-1] );

			if( vs[0] == null ) //scalar input
				vs[0] = _scalarStats;
			if( vs[2] == null ) //scalar output
				vs[2] = _scalarStats;
		}
		else if( inst instanceof MultiReturnBuiltinCPInstruction ) {
			//applies to qr, lu, eigen (cost computation on input1)
			MultiReturnBuiltinCPInstruction minst = (MultiReturnBuiltinCPInstruction) inst;
			vs[0] = stats.get( minst.input1.getName() );
			vs[1] = stats.get( minst.getOutput(0).getName() );
			vs[2] = stats.get( minst.getOutput(1).getName() );
		}
		else if( inst instanceof VariableCPInstruction ) {
			setUnknownStats(vs);

			VariableCPInstruction varinst = (VariableCPInstruction) inst;
			if( varinst.getOpcode().equals("write") ) {
				//special handling write of matrix objects (non existing if scalar)
				if( stats.containsKey( varinst.getInput1().getName() ) )
					vs[0] = stats.get( varinst.getInput1().getName() );
				attr = new String[]{varinst.getInput3().getName()};
			}
		}
		else {
			setUnknownStats(vs);
		}

		//maintain var status (CP output always inmem)
		vs[2]._inmem = true;

		ret[0] = vs;
		ret[1] = attr;

		return ret;
	}
}

