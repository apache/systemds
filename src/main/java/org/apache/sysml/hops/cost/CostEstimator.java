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

package org.apache.sysml.hops.cost;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringTokenizer;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.ExternalFunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.MRInstructionParser;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.instructions.cp.AggregateTernaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPInstruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysml.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysml.runtime.instructions.cp.StringInitCPInstruction;
import org.apache.sysml.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysml.runtime.instructions.mr.MRInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.operators.CMOperator;
import org.apache.sysml.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysml.runtime.util.UtilFunctions;

public abstract class CostEstimator 
{
	
	protected static final Log LOG = LogFactory.getLog(CostEstimator.class.getName());
	
	private static final int DEFAULT_NUMITER = 15;
	
	protected static final VarStats _unknownStats = new VarStats(1,1,-1,-1,-1,false);
	protected static final VarStats _scalarStats = new VarStats(1,1,1,1,1,true);
	
	/**
	 * 
	 * @param rtprog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public double getTimeEstimate(Program rtprog, LocalVariableMap vars, HashMap<String,VarStats> stats) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		double costs = 0;

		//obtain stats from symboltable (e.g., during recompile)
		maintainVariableStatistics(vars, stats);
						
		//get cost estimate
		for( ProgramBlock pb : rtprog.getProgramBlocks() )
			costs += rGetTimeEstimate(pb, stats, new HashSet<String>(), true);
		
		return costs;
	}
	
	/**
	 * 
	 * @param pb
	 * @param vars
	 * @param stats
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public double getTimeEstimate(ProgramBlock pb, LocalVariableMap vars, HashMap<String,VarStats> stats, boolean recursive) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//obtain stats from symboltable (e.g., during recompile)
		maintainVariableStatistics(vars, stats);
				
		//get cost estimate
		return rGetTimeEstimate(pb, stats, new HashSet<String>(), recursive);
	}
	
		
	/**
	 * 
	 * @param hops
	 * @param vars
	 * @return
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 * @throws LopsException 
	 * @throws HopsException 
	 */
	public double getTimeEstimate( ArrayList<Hop> hops, LocalVariableMap vars, HashMap<String,VarStats> stats ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, HopsException, LopsException, IOException
	{
		double costs = 0;
		
		ArrayList<Instruction> linst = Recompiler.recompileHopsDag(null, hops, vars, null, false, 0);
		ProgramBlock pb = new ProgramBlock(null);
		pb.setInstructions(linst);
		
		//obtain stats from symboltable (e.g., during recompile)
		maintainVariableStatistics(vars, stats);
		
		//get cost estimate
		costs = rGetTimeEstimate(pb, stats, new HashSet<String>(), true);
		
		return costs;
	}
	
	/**
	 * 
	 * @param pb
	 * @param vars
	 * @param stats
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private double rGetTimeEstimate(ProgramBlock pb, HashMap<String,VarStats> stats, HashSet<String> memoFunc, boolean recursive) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		double ret = 0;
		
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			if( recursive )
				for (ProgramBlock pb2 : tmp.getChildBlocks())
					ret += rGetTimeEstimate(pb2, stats, memoFunc, recursive);
			ret *= DEFAULT_NUMITER;
		}
		else if (pb instanceof IfProgramBlock)
		{
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
		else if (pb instanceof ForProgramBlock) //includes ParFORProgramBlock
		{ 
			ForProgramBlock tmp = (ForProgramBlock)pb;	
			if( recursive )
				for( ProgramBlock pb2 : tmp.getChildBlocks() )
					ret += rGetTimeEstimate(pb2, stats, memoFunc, recursive);
			
			ret *= getNumIterations(stats, tmp.getIterablePredicateVars());			
		}		
		else if ( pb instanceof FunctionProgramBlock 
				  && !(pb instanceof ExternalFunctionProgramBlock)) //see generic
		{
			FunctionProgramBlock tmp = (FunctionProgramBlock) pb;
			if( recursive )
				for( ProgramBlock pb2 : tmp.getChildBlocks() )
					ret += rGetTimeEstimate(pb2, stats, memoFunc, recursive);
		}
		else 
		{	
			ArrayList<Instruction> tmp = pb.getInstructions();
			
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
				else if(inst instanceof MRJobInstruction) //MR
				{
					//obtain stats for job
					maintainMRJobInstVariableStatistics(inst, stats);
					
					//extract input statistics
					Object[] o = extractMRJobInstStatistics(inst, stats);
					VarStats[] vs = (VarStats[]) o[0];

					//if(LOG.isDebugEnabled())
					//	LOG.debug(inst);
					
					if(LOG.isDebugEnabled())
						LOG.debug("Begin MRJob type="+((MRJobInstruction)inst).getJobType());
					
					//call time estimation for complex MR inst
					ret += getMRJobInstTimeEstimate(inst, vs, null);
					
					if(LOG.isDebugEnabled())
						LOG.debug("End MRJob");
					
					//cleanup stats for job
					cleanupMRJobVariableStatistics(inst, stats);
				}
			}
		}
		
		return ret;
	}
	
	
	/**
	 * 
	 * @param vars
	 * @param stats
	 * @throws DMLRuntimeException 
	 */
	private void maintainVariableStatistics( LocalVariableMap vars, HashMap<String, VarStats> stats ) 
		throws DMLRuntimeException
	{
		for( String varname : vars.keySet() )
		{
			Data dat = vars.get(varname);
			VarStats vs = null;
			if( dat instanceof MatrixObject ) //matrix
			{
				MatrixObject mo = (MatrixObject) dat;
				MatrixCharacteristics mc = ((MatrixDimensionsMetaData)mo.getMetaData()).getMatrixCharacteristics();
				long rlen = mc.getRows();
				long clen = mc.getCols();
				long brlen = mc.getRowsPerBlock();
				long bclen = mc.getColsPerBlock();
				long nnz = mc.getNonZeros();
				boolean inmem = mo.getStatusAsString().equals("CACHED");
				vs = new VarStats(rlen, clen, brlen, bclen, nnz, inmem);
			}
			else //scalar
			{
				vs = _scalarStats; 
			}
			
			stats.put(varname, vs);
			//System.out.println(varname+" "+vs);
		}
	}
	
	/**
	 * 
	 * @param inst
	 * @param stats
	 */
	private void maintainCPInstVariableStatistics( CPInstruction inst, HashMap<String, VarStats> stats )
	{
		if( inst instanceof VariableCPInstruction )
		{
			String optype = inst.getOpcode();
			String[] parts = InstructionUtils.getInstructionParts(inst.toString());
			
			if( optype.equals("createvar") ) {
				if( parts.length<10 )
					return;
				String varname = parts[1];
				long rlen = Long.parseLong(parts[5]);
				long clen = Long.parseLong(parts[6]);
				long brlen = Long.parseLong(parts[7]);
				long bclen = Long.parseLong(parts[8]);
				long nnz = Long.parseLong(parts[9]);
				VarStats vs = new VarStats(rlen, clen, brlen, bclen, nnz, false);
				stats.put(varname, vs);
				
				//System.out.println(varname+" "+vs);
			}
			else if ( optype.equals("cpvar") ) {
				String varname = parts[1];
				String varname2 = parts[2];
				VarStats vs = stats.get(varname);
				stats.put(varname2, vs);
			}
			else if ( optype.equals("mvvar") ) {
				String varname = parts[1];
				String varname2 = parts[2];
				VarStats vs = stats.remove(varname);
				stats.put(varname2, vs);
			}
			else if( optype.equals("rmvar") ) {
				String varname = parts[1];
				stats.remove(varname);
			}
		}	
		else if( inst instanceof DataGenCPInstruction ){
			DataGenCPInstruction randInst = (DataGenCPInstruction) inst;
			String varname = randInst.output.getName();
			long rlen = randInst.getRows();
			long clen = randInst.getCols();
			long brlen = randInst.getRowsInBlock();
			long bclen = randInst.getColsInBlock();
			long nnz = (long) (randInst.getSparsity() * rlen * clen);
			VarStats vs = new VarStats(rlen, clen, brlen, bclen, nnz, true);
			stats.put(varname, vs);
		}
		else if( inst instanceof StringInitCPInstruction ){
			StringInitCPInstruction iinst = (StringInitCPInstruction) inst;
			String varname = iinst.output.getName();
			long rlen = iinst.getRows();
			long clen = iinst.getCols();
			VarStats vs = new VarStats(rlen, clen, ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize(), rlen*clen, true);
			stats.put(varname, vs);	
		}
		else if( inst instanceof FunctionCallCPInstruction )
		{
			FunctionCallCPInstruction finst = (FunctionCallCPInstruction) inst;
			ArrayList<String> outVars = finst.getBoundOutputParamNames();
			for( String varname : outVars )
			{
				stats.put(varname, _unknownStats);
				//System.out.println(varname+" "+vs);
			}
		}
	}

	/**
	 * 
	 * @param inst
	 * @param stats
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	private void maintainMRJobInstVariableStatistics( Instruction inst, HashMap<String, VarStats> stats ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		MRJobInstruction jobinst = (MRJobInstruction)inst;
		
		//input sizes (varname, index mapping)
		String[] inVars = jobinst.getInputVars();
		int index = -1;
		for( String varname : inVars )
		{
			VarStats vs = stats.get(varname);
			if( vs==null )
				vs = _unknownStats;
			stats.put(String.valueOf(++index), vs);
		}
		
		//rand output
		String rdInst = jobinst.getIv_randInstructions();
		if( rdInst != null && rdInst.length()>0 )
		{
			StringTokenizer st = new StringTokenizer(rdInst,Lop.INSTRUCTION_DELIMITOR);
			while( st.hasMoreTokens() ) //foreach rand instruction
			{				
				String[] parts = InstructionUtils.getInstructionParts(st.nextToken());
				byte outIndex = Byte.parseByte(parts[2]);
				long rlen = parts[3].contains(Lop.VARIABLE_NAME_PLACEHOLDER)?-1:UtilFunctions.parseToLong(parts[3]);
				long clen = parts[4].contains(Lop.VARIABLE_NAME_PLACEHOLDER)?-1:UtilFunctions.parseToLong(parts[4]);
				long brlen = Long.parseLong(parts[5]);
				long bclen = Long.parseLong(parts[6]);
				long nnz = (long) (Double.parseDouble(parts[9]) * rlen * clen);
				VarStats vs = new VarStats(rlen, clen, brlen, bclen, nnz, false);
				stats.put(String.valueOf(outIndex), vs);	
			}
		}
		
		//compute intermediate result indices
		HashMap<Byte,MatrixCharacteristics> dims = new HashMap<Byte, MatrixCharacteristics>();
		//populate input indices
		for( Entry<String,VarStats> e : stats.entrySet() )
		{
			if(UtilFunctions.isIntegerNumber(e.getKey()))
			{
				byte ix = Byte.parseByte(e.getKey());
				VarStats vs = e.getValue();
				if( vs !=null )
				{
					MatrixCharacteristics mc = new MatrixCharacteristics(vs._rlen, vs._clen, (int)vs._brlen, (int)vs._bclen, (long)vs._nnz);
					dims.put(ix, mc);
				}
			}
		}
		//compute dims for all instructions
		String[] instCat = new String[]{
				jobinst.getIv_randInstructions(),
				jobinst.getIv_recordReaderInstructions(),
				jobinst.getIv_instructionsInMapper(),
				jobinst.getIv_shuffleInstructions(),
				jobinst.getIv_aggInstructions(),
				jobinst.getIv_otherInstructions()};		
		for( String linstCat : instCat )
			if( linstCat !=null && linstCat.length()>0 )
			{
				String[] linst = linstCat.split(Instruction.INSTRUCTION_DELIM);
				for( String instStr : linst )
				{
					String instStr2 = replaceInstructionPatch(instStr);
					MRInstruction mrinst = MRInstructionParser.parseSingleInstruction(instStr2);
					MatrixCharacteristics.computeDimension(dims, mrinst);
				}
			}
		
		//create varstats if necessary
		for( Entry<Byte,MatrixCharacteristics> e : dims.entrySet() )
		{
			byte ix = e.getKey();
			if( !stats.containsKey(String.valueOf(ix)) )
			{
				MatrixCharacteristics mc = e.getValue();
				VarStats vs = new VarStats(mc.getRows(), mc.getCols(), mc.getRowsPerBlock(), mc.getColsPerBlock(), mc.getNonZeros(), false);
				stats.put(String.valueOf(ix), vs);	
			}
		}
		
		//map result indexes
		String[] outLabels = jobinst.getOutputVars();
		byte[] resultIndexes = jobinst.getIv_resultIndices();
		for( int i=0; i<resultIndexes.length; i++ )
		{
			String varname = outLabels[i];
			VarStats varvs = stats.get(String.valueOf(resultIndexes[i]));
			if( varvs==null )
			{
				varvs = stats.get(outLabels[i]);
			}
			varvs._inmem = false;
			stats.put(varname, varvs);
		}	
	}
	
	/**
	 * 
	 * @param inst
	 * @return
	 */
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
	
	/**
	 * 
	 * @param inst
	 * @param stats
	 * @return
	 */
	private Object[] extractCPInstStatistics( Instruction inst, HashMap<String, VarStats> stats )
	{		
		Object[] ret = new Object[2]; //stats, attrs
		VarStats[] vs = new VarStats[3];
		String[] attr = null; 

		if( inst instanceof UnaryCPInstruction )
		{
			if( inst instanceof DataGenCPInstruction )
			{
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
			else if( inst instanceof StringInitCPInstruction )
			{
				StringInitCPInstruction rinst = (StringInitCPInstruction) inst;
				vs[0] = _unknownStats;
				vs[1] = _unknownStats;
				vs[2] = stats.get( rinst.output.getName() );
			}
			else //general unary
			{
				UnaryCPInstruction uinst = (UnaryCPInstruction) inst;
				vs[0] = stats.get( uinst.input1.getName() );
				vs[1] = _unknownStats;
				vs[2] = stats.get( uinst.output.getName() );
				
				if( vs[0] == null ) //scalar input, e.g., print
					vs[0] = _scalarStats;
				if( vs[2] == null ) //scalar output
					vs[2] = _scalarStats;
				
				if( inst instanceof MMTSJCPInstruction )
				{
					String type = ((MMTSJCPInstruction)inst).getMMTSJType().toString();
					attr = new String[]{type};
				} 
				else if( inst instanceof AggregateUnaryCPInstruction )
				{
					String[] parts = InstructionUtils.getInstructionParts(inst.toString());
					String opcode = parts[0];
					if( opcode.equals("cm") )
						attr = new String[]{parts[parts.length-2]};						
				} 
			}
		}
		else if( inst instanceof BinaryCPInstruction )
		{
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
		else if( inst instanceof AggregateTernaryCPInstruction )
		{
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
		else if( inst instanceof ParameterizedBuiltinCPInstruction )
		{
			//ParameterizedBuiltinCPInstruction pinst = (ParameterizedBuiltinCPInstruction) inst;
			String[] parts = InstructionUtils.getInstructionParts(inst.toString());
			String opcode = parts[0];
			if( opcode.equals("groupedagg") )
			{				
				HashMap<String,String> paramsMap = ParameterizedBuiltinCPInstruction.constructParameterMap(parts);
				String fn = paramsMap.get("fn");
				String order = paramsMap.get("order");
				AggregateOperationTypes type = CMOperator.getAggOpType(fn, order);
				attr = new String[]{String.valueOf(type.ordinal())};
			}
			else if( opcode.equals("rmempty") )
			{
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
		else if( inst instanceof MultiReturnBuiltinCPInstruction )
		{
			//applies to qr, lu, eigen (cost computation on input1)
			MultiReturnBuiltinCPInstruction minst = (MultiReturnBuiltinCPInstruction) inst;
			vs[0] = stats.get( minst.input1.getName() );
			vs[1] = stats.get( minst.getOutput(0).getName() );
			vs[2] = stats.get( minst.getOutput(1).getName() );
		}
		else if( inst instanceof VariableCPInstruction )
		{
			setUnknownStats(vs);
			
			VariableCPInstruction varinst = (VariableCPInstruction) inst;
			if( varinst.getOpcode().equals("write") ) {
				//special handling write of matrix objects (non existing if scalar)
				if( stats.containsKey( varinst.getInput1().getName() ) )
					vs[0] = stats.get( varinst.getInput1().getName() );	
				attr = new String[]{varinst.getInput3().getName()};
			}	
		}
		else
		{
			setUnknownStats(vs);		
		}
		
		//maintain var status (CP output always inmem)
		vs[2]._inmem = true;
		
		ret[0] = vs;
		ret[1] = attr;
		
		return ret;
	}
	
	private void setUnknownStats(VarStats[] vs) {
		vs[0] = _unknownStats;
		vs[1] = _unknownStats;
		vs[2] = _unknownStats;	
	}
	
	/**
	 * 
	 * @param inst
	 * @param stats
	 * @return
	 */
	private Object[] extractMRJobInstStatistics( Instruction inst, HashMap<String, VarStats> stats )
	{
		Object[] ret = new Object[2]; //stats, attrs
		VarStats[] vs = null;
		String[] attr = null; 
		
		MRJobInstruction jinst = (MRJobInstruction)inst;
		
		//get number of indices 
		byte[] indexes = jinst.getIv_resultIndices();
		byte maxIx = -1;
		for( int i=0; i<indexes.length; i++ )
			if( maxIx < indexes[i] )
				maxIx = indexes[i];
		
		vs = new VarStats[ maxIx+1 ];
		
		//get inputs, intermediates, and outputs
		for( int i=0; i<vs.length; i++ ){
			vs[i] = stats.get(String.valueOf(i));
			if( vs[i]==null )
			{
				vs[i] = _unknownStats;
			}
		}
		
		//result preparation
		ret[0] = vs;
		ret[1] = attr;
		
		return ret;
	}	
	
	/**
	 * 
	 * @param inst
	 * @param stats
	 */
	private void cleanupMRJobVariableStatistics( Instruction inst, HashMap<String, VarStats> stats )
	{
		MRJobInstruction jinst = (MRJobInstruction)inst;
		
		//get number of indices 
		byte[] indexes = jinst.getIv_resultIndices();
		byte maxIx = -1;
		for( int i=0; i<indexes.length; i++ )
			if( maxIx < indexes[i] )
				maxIx = indexes[i];
		
		//remove all stats up to max index
		for( int i=0; i<=maxIx; i++ )
		{
			VarStats tmp = stats.remove(String.valueOf(i));
			if( tmp!=null )
				tmp._inmem = false; //all MR job outptus on HDFS
		}
	}	
	
	/**
	 * TODO use of vars - needed for recompilation
	 * TODO w/o exception
	 * 
	 * @param vars
	 * @param stats
	 * @param pred
	 * @return
	 */
	private int getNumIterations(HashMap<String,VarStats> stats, String[] pred)
	{
		int N = DEFAULT_NUMITER;
		if( pred != null && pred[1]!=null && pred[2]!=null && pred[3]!=null )
		{
			try 
			{
				int from = Integer.parseInt(pred[1]);
				int to = Integer.parseInt(pred[2]);
				int increment = Integer.parseInt(pred[3]);
				N = (int)Math.ceil(((double)(to-from+1))/increment);
			}
			catch(Exception ex){}
		}           
		return N;              
	}
	
	
	/**
	 * 
	 * @param inst
	 * @param vs
	 * @param args
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected abstract double getCPInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args  ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException;
	
	/**
	 * 
	 * @param inst
	 * @param vs
	 * @param args
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected abstract double getMRJobInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args  ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException;
}
