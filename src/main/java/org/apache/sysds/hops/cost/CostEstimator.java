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

package org.apache.sysds.hops.cost;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData.CacheStatus;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.AggregateTernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.StringInitCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public abstract class CostEstimator
{
	protected static final Log LOG = LogFactory.getLog(CostEstimator.class.getName());
	
	private static final int DEFAULT_NUMITER = 15;
	
	protected static final VarStats _unknownStats = new VarStats(1,1,-1,-1,false);
	protected static final VarStats _scalarStats = new VarStats(1,1,1,1,true);
	
	public double getTimeEstimate(Program rtprog, LocalVariableMap vars, HashMap<String,VarStats> stats) {
		double costs = 0;

		//obtain stats from symboltable (e.g., during recompile)
		maintainVariableStatistics(vars, stats);
						
		//get cost estimate
		for( ProgramBlock pb : rtprog.getProgramBlocks() )
			costs += rGetTimeEstimate(pb, stats, new HashSet<String>(), true);
		
		return costs;
	}
	
	public double getTimeEstimate(ProgramBlock pb, LocalVariableMap vars, HashMap<String,VarStats> stats, boolean recursive) {
		//obtain stats from symboltable (e.g., during recompile)
		maintainVariableStatistics(vars, stats);
		
		//get cost estimate
		return rGetTimeEstimate(pb, stats, new HashSet<String>(), recursive);
	}

	private double rGetTimeEstimate(ProgramBlock pb, HashMap<String,VarStats> stats, HashSet<String> memoFunc, boolean recursive) {
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
					Object[] o = extractCPInstStatistics(inst, stats);
					VarStats[] vs = (VarStats[]) o[0];
					String[] attr = (String[]) o[1];
					
					//if(LOG.isDebugEnabled())
					//	LOG.debug(inst);
					
					//call time estimation for inst
					ret += getCPInstTimeEstimate(inst, vs, attr); 
					
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
				}
			}
		}
		
		return ret;
	}
	
	private static void maintainVariableStatistics( LocalVariableMap vars, HashMap<String, VarStats> stats ) {
		for( String varname : vars.keySet() )
		{
			Data dat = vars.get(varname);
			VarStats vs = null;
			if( dat instanceof MatrixObject ) //matrix
			{
				MatrixObject mo = (MatrixObject) dat;
				DataCharacteristics dc = mo.getDataCharacteristics();
				long rlen = dc.getRows();
				long clen = dc.getCols();
				int blen = dc.getBlocksize();
				long nnz = dc.getNonZeros();
				boolean inmem = mo.getStatus()==CacheStatus.CACHED;
				vs = new VarStats(rlen, clen, blen, nnz, inmem);
			}
			else //scalar
			{
				vs = _scalarStats; 
			}
			
			stats.put(varname, vs);
		}
	}
	
	private static void maintainCPInstVariableStatistics( CPInstruction inst, HashMap<String, VarStats> stats )
	{
		if( inst instanceof VariableCPInstruction )
		{
			String optype = inst.getOpcode();
			String[] parts = InstructionUtils.getInstructionParts(inst.toString());
			
			if( optype.equals(Opcodes.CREATEVAR.toString()) ) {
				if( parts.length<10 )
					return;
				String varname = parts[1];
				long rlen = Long.parseLong(parts[6]);
				long clen = Long.parseLong(parts[7]);
				int blen = Integer.parseInt(parts[8]);
				long nnz = Long.parseLong(parts[9]);
				VarStats vs = new VarStats(rlen, clen, blen, nnz, false);
				stats.put(varname, vs);
			}
			else if ( optype.equals(Opcodes.CPVAR.toString()) ) {
				String varname = parts[1];
				String varname2 = parts[2];
				VarStats vs = stats.get(varname);
				stats.put(varname2, vs);
			}
			else if ( optype.equals(Opcodes.MVVAR.toString()) ) {
				String varname = parts[1];
				String varname2 = parts[2];
				VarStats vs = stats.remove(varname);
				stats.put(varname2, vs);
			}
			else if( optype.equals(Opcodes.RMVAR.toString()) ) {
				String varname = parts[1];
				stats.remove(varname);
			}
		}	
		else if( inst instanceof DataGenCPInstruction ){
			DataGenCPInstruction randInst = (DataGenCPInstruction) inst;
			String varname = randInst.output.getName();
			long rlen = randInst.getRows();
			long clen = randInst.getCols();
			int blen = randInst.getBlocksize();
			long nnz = (long) (randInst.getSparsity() * rlen * clen);
			VarStats vs = new VarStats(rlen, clen, blen, nnz, true);
			stats.put(varname, vs);
		}
		else if( inst instanceof StringInitCPInstruction ){
			StringInitCPInstruction iinst = (StringInitCPInstruction) inst;
			String varname = iinst.output.getName();
			long rlen = iinst.getRows();
			long clen = iinst.getCols();
			VarStats vs = new VarStats(rlen, clen, ConfigurationManager.getBlocksize(), rlen*clen, true);
			stats.put(varname, vs);	
		}
		else if( inst instanceof FunctionCallCPInstruction )
		{
			FunctionCallCPInstruction finst = (FunctionCallCPInstruction) inst;
			for( String varname : finst.getBoundOutputParamNames() )
				stats.put(varname, _unknownStats);
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
	
	private static Object[] extractCPInstStatistics( Instruction inst, HashMap<String, VarStats> stats )
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
					if( opcode.equals(Opcodes.CM.toString()) )
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
			if( opcode.equals(Opcodes.GROUPEDAGG.toString()) ) {
				HashMap<String,String> paramsMap = ParameterizedBuiltinCPInstruction.constructParameterMap(parts);
				String fn = paramsMap.get("fn");
				String order = paramsMap.get("order");
				AggregateOperationTypes type = CMOperator.getAggOpType(fn, order);
				attr = new String[]{String.valueOf(type.ordinal())};
			}
			else if( opcode.equals(Opcodes.RMEMPTY.toString()) ) {
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
			if( varinst.getOpcode().equals(Opcodes.WRITE.toString()) ) {
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
	
	private static void setUnknownStats(VarStats[] vs) {
		vs[0] = _unknownStats;
		vs[1] = _unknownStats;
		vs[2] = _unknownStats;
	}
		
	private static long getNumIterations(ForProgramBlock pb) {
		return OptimizerUtils.getNumIterations(pb, DEFAULT_NUMITER);
	}

	protected abstract double getCPInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args );
}
