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

package org.apache.sysml.yarn.ropt;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.cost.CostEstimationWrapper;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptTreeConverter;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.yarn.DMLYarnClient;
import org.apache.sysml.yarn.ropt.YarnOptimizerUtils.GridEnumType;

/**
 * TODO parallel version with exposed numThreads parameter
 * 
 */
public class ResourceOptimizer 
{
	
	private static final Log LOG = LogFactory.getLog(ResourceOptimizer.class);
	
	//internal configuration parameters 
	public static final long MIN_CP_BUDGET = 512*1024*1024; //512MB
	public static final boolean INCLUDE_PREDICATES = true;
	public static final boolean PRUNING_SMALL = true;
	public static final boolean PRUNING_UNKNOWN = true;
	public static final boolean COSTS_MAX_PARALLELISM = true;
	public static final boolean COST_INDIVIDUAL_BLOCKS = true;
	
	private static long _cntCompilePB = 0;
	private static long _cntCostPB = 0;
	
	public synchronized static ResourceConfig optimizeResourceConfig( ArrayList<ProgramBlock> prog, YarnClusterConfig cc, GridEnumType cptype, GridEnumType mrtype ) 
		throws DMLRuntimeException
	{
		ResourceConfig ROpt = null;
		
		try
		{
			//init statistics and counters
			Timing time = new Timing(true);
			initStatistics();
			
			//get constraints (yarn-specific: force higher min to limit degree of parallelism)
			long max = (long)(YarnOptimizerUtils.toB(cc.getMaxAllocationMB()) / DMLYarnClient.MEM_FACTOR);
			long minCP = (long) Math.max(YarnOptimizerUtils.toB(cc.getMinAllocationMB()) / DMLYarnClient.MEM_FACTOR, MIN_CP_BUDGET);
			long minMR = YarnOptimizerUtils.computeMinContraint(minCP, max, cc.getAvgNumCores());
			
			//enumerate grid points for given types (refers to jvm max heap)
			ArrayList<Long> SRc = enumerateGridPoints(prog, minCP, max, cptype);
			ArrayList<Long> SRm = enumerateGridPoints(prog, minMR, max, mrtype);
			
			//init resource config and global costs
			ROpt = new ResourceConfig(prog, minMR);
			double costOpt = Double.MAX_VALUE;
			
			for( Long rc : SRc ) //enumerate CP memory rc
			{
				//baseline compile and pruning
				ArrayList<ProgramBlock> B = compileProgram(prog, null, rc, minMR); //unrolled Bp
				ArrayList<ProgramBlock> Bp = pruneProgramBlocks( B );
				LOG.debug("Enum (rc="+rc+"): |B|="+B.size()+", |Bp|="+Bp.size());
				
				//init local memo table [resource, cost]
				double[][] memo = initLocalMemoTable( Bp, minMR );
				
				for( int i=0; i<Bp.size(); i++ ) //for all relevant blocks
				{
					ProgramBlock pb = Bp.get(i);
					
					for( Long rm : SRm ) //for each MR memory 
					{
						//recompile program block 
						recompileProgramBlock(pb, rc, rm);
						
						//local costing and memo table maintenance (cost entire program to account for 
						//in-memory status of variables and loops)
						double lcost = getProgramCosts( pb );
						if( lcost < memo[i][1] ) { //accept new local opt
							memo[i][0] = rm;
							memo[i][1] = lcost;
							//LOG.debug("Enum (rc="+rc+"): found new local opt w/ cost="+lcost);			
 						}
						//LOG.debug("Enum (rc="+rc+", rm="+rm+"): lcost="+lcost+", mincost="+memo[i][1]);						
					}
				}			
				
				//global costing 
				double[][] gmemo = initGlobalMemoTable(B, Bp, memo, minMR);
				recompileProgramBlocks(B, rc, gmemo);
				double gcost = getProgramCosts(B.get(0).getProgram());
				if( gcost < costOpt ){ //accept new global opt
					ROpt.setCPResource(rc.longValue());
					ROpt.setMRResources(B, gmemo);
					costOpt = gcost;
					LOG.debug("Enum (rc="+rc+"): found new opt w/ cost="+gcost);
				}
			}	

			//print optimization summary
			LOG.info("Optimization summary:");
			LOG.info("-- optimal plan (rc, rm): "+YarnOptimizerUtils.toMB(ROpt.getCPResource())+"MB, "+YarnOptimizerUtils.toMB(ROpt.getMaxMRResource())+"MB");
			LOG.info("-- costs of optimal plan: "+costOpt);
			LOG.info("-- # of block compiles:   "+_cntCompilePB);
			LOG.info("-- # of block costings:   "+_cntCostPB);
			LOG.info("-- optimization time:     "+String.format("%.3f", (double)time.stop()/1000)+" sec.");
			LOG.info("-- optimal plan details:  "+ROpt.serialize());
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
	
		return ROpt;
	}

	public static ArrayList<ProgramBlock> compileProgram( ArrayList<ProgramBlock> prog, ResourceConfig rc ) 
		throws DMLRuntimeException, HopsException, LopsException, IOException
	{
		//recompile program block hierarchy to list of blocks and apply optimized resource configuration
		ArrayList<ProgramBlock> B = compileProgram(prog, null, rc.getCPResource(), rc.getMaxMRResource());
		ResourceOptimizer.recompileProgramBlocks(B, rc.getCPResource(), rc.getMRResourcesMemo());
		
		return B;
	}

	private static ArrayList<ProgramBlock> compileProgram( ArrayList<ProgramBlock> prog, ArrayList<ProgramBlock> B, double cp, double mr ) 
		throws DMLRuntimeException, HopsException, LopsException, IOException
	{
		if( B == null ) //init 
		{
			B = new ArrayList<ProgramBlock>();
			
			InfrastructureAnalyzer.setLocalMaxMemory( (long)cp );
			InfrastructureAnalyzer.setRemoteMaxMemoryMap( (long)mr );
			InfrastructureAnalyzer.setRemoteMaxMemoryReduce( (long)mr );	
			OptimizerUtils.resetDefaultSize(); //dependent on cp, mr
		}
		
		for( ProgramBlock pb : prog )
			compileProgram( pb, B, cp, mr );
		
		return B;
	}

	private static ArrayList<ProgramBlock> compileProgram( ProgramBlock pb, ArrayList<ProgramBlock> B, double cp, double mr ) 
		throws DMLRuntimeException, HopsException, LopsException, IOException
	{
		if (pb instanceof FunctionProgramBlock)
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			compileProgram(fpb.getChildBlocks(), B, cp, mr);
		}
		else if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock)pb;
			WhileStatementBlock sb = (WhileStatementBlock) pb.getStatementBlock();
			if( INCLUDE_PREDICATES && sb!=null && sb.getPredicateHops()!=null ){
				ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
					sb.getPredicateHops(), new LocalVariableMap(), null, false, false, 0);
				wpb.setPredicate( inst );
				B.add(wpb);
				_cntCompilePB ++;
			}				
			compileProgram(wpb.getChildBlocks(), B, cp, mr);
		}	
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock)pb;
			IfStatementBlock sb = (IfStatementBlock) ipb.getStatementBlock();
			if( INCLUDE_PREDICATES && sb!=null && sb.getPredicateHops()!=null ){
				ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
					sb.getPredicateHops(), new LocalVariableMap(), null, false, false, 0);
				ipb.setPredicate( inst );
				B.add(ipb);
				_cntCompilePB ++;
			}
			compileProgram(ipb.getChildBlocksIfBody(), B, cp, mr);
			compileProgram(ipb.getChildBlocksElseBody(), B, cp, mr);
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock)pb;
			ForStatementBlock sb = (ForStatementBlock) fpb.getStatementBlock();
			if( INCLUDE_PREDICATES && sb!=null ){
				if( sb.getFromHops()!=null ){
					ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
						sb.getFromHops(), new LocalVariableMap(), null, false, false, 0);
					fpb.setFromInstructions( inst );	
				}
				if( sb.getToHops()!=null ){
					ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
						sb.getToHops(), new LocalVariableMap(), null, false, false, 0);
					fpb.setToInstructions( inst );	
				}
				if( sb.getIncrementHops()!=null ){
					ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
						sb.getIncrementHops(), new LocalVariableMap(), null, false, false, 0);
					fpb.setIncrementInstructions( inst );	
				}
				B.add(fpb);
				_cntCompilePB ++;
			}
			compileProgram(fpb.getChildBlocks(), B, cp, mr);
		}
		else
		{
			StatementBlock sb = pb.getStatementBlock();
			ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
				sb, sb.get_hops(), new LocalVariableMap(), null, false, false, 0);
			pb.setInstructions( inst );
			B.add(pb);
			_cntCompilePB ++;
		}
		
		return B;
	}

	private static void recompileProgramBlocks( ArrayList<ProgramBlock> pbs, long cp, double[][] memo ) 
		throws DMLRuntimeException, HopsException, LopsException, IOException
	{
		for( int i=0; i<pbs.size(); i++ )
		{
			ProgramBlock pb = pbs.get(i);
			long mr = (long)memo[i][0];
			recompileProgramBlock(pb, cp, mr);
		}
	}

	private static void recompileProgramBlock( ProgramBlock pb, long cp, long mr ) 
		throws DMLRuntimeException, HopsException, LopsException, IOException
	{
		//init compiler memory budget
		InfrastructureAnalyzer.setLocalMaxMemory( cp );
		InfrastructureAnalyzer.setRemoteMaxMemoryMap( mr );
		InfrastructureAnalyzer.setRemoteMaxMemoryReduce( mr );
		OptimizerUtils.resetDefaultSize(); //dependent on cp, mr
		
		//recompile instructions (incl predicates)
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock)pb;
			WhileStatementBlock sb = (WhileStatementBlock) pb.getStatementBlock();
			if( INCLUDE_PREDICATES && sb!=null && sb.getPredicateHops()!=null ){
				ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
					sb.getPredicateHops(), new LocalVariableMap(), null, false, false, 0);
				inst = annotateMRJobInstructions(inst, cp, mr);
				wpb.setPredicate( inst );
			}				
		}	
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock)pb;
			IfStatementBlock sb = (IfStatementBlock) ipb.getStatementBlock();
			if( INCLUDE_PREDICATES && sb!=null && sb.getPredicateHops()!=null ){
				ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
					sb.getPredicateHops(), new LocalVariableMap(), null, false, false, 0);
				inst = annotateMRJobInstructions(inst, cp, mr);
				ipb.setPredicate( inst );
			}
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock)pb;
			ForStatementBlock sb = (ForStatementBlock) fpb.getStatementBlock();
			if( INCLUDE_PREDICATES && sb!=null ){
				if( sb.getFromHops()!=null ){
					ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
						sb.getFromHops(), new LocalVariableMap(), null, false, false, 0);
					inst = annotateMRJobInstructions(inst, cp, mr);
					fpb.setFromInstructions( inst );	
				}
				if( sb.getToHops()!=null ){
					ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
						sb.getToHops(), new LocalVariableMap(), null, false, false, 0);
					inst = annotateMRJobInstructions(inst, cp, mr);
					fpb.setToInstructions( inst );	
				}
				if( sb.getIncrementHops()!=null ){
					ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
						sb.getIncrementHops(), new LocalVariableMap(), null, false, false, 0);
					inst = annotateMRJobInstructions(inst, cp, mr);
					fpb.setIncrementInstructions( inst );	
				}
			}
		}
		else //last-level program blocks
		{
			StatementBlock sb = pb.getStatementBlock();
			ArrayList<Instruction> inst = Recompiler.recompileHopsDag(
				sb, sb.get_hops(), new LocalVariableMap(), null, false, false, 0);
			inst = annotateMRJobInstructions(inst, cp, mr);
			pb.setInstructions( inst );
		}
		
		_cntCompilePB ++;
	}

	private static ArrayList<Instruction> annotateMRJobInstructions( ArrayList<Instruction> inst, long cp, long mr ) 
		throws DMLRuntimeException
	{
		//check for empty instruction lists (e.g., predicates)
		if( inst == null || !COSTS_MAX_PARALLELISM )
			return inst;
		
		try
		{
			for( int i=0; i<inst.size(); i++ )
			{
				Instruction linst = inst.get(i);
				if( linst instanceof MRJobInstruction ){
					//copy mr job instruction
					MRJobResourceInstruction newlinst = new MRJobResourceInstruction((MRJobInstruction)linst);
					
					//compute and annotate
					long maxMemPerNode = (long)YarnClusterAnalyzer.getMaxAllocationBytes();
					long nNodes = YarnClusterAnalyzer.getNumNodes();
					long totalMem = nNodes * maxMemPerNode;
					long maxMRTasks =   (long)(totalMem - DMLYarnClient.computeMemoryAllocation(cp)) 
							          / (long)DMLYarnClient.computeMemoryAllocation(mr);
					newlinst.setMaxMRTasks( maxMRTasks );
					
					//write enhanced instruction back
					inst.set(i, newlinst);
				}
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
		
		return inst;
	}

	private static double getProgramCosts( ProgramBlock pb ) 
		throws DMLRuntimeException, HopsException
	{
		double val = 0;
		if( COST_INDIVIDUAL_BLOCKS ) {
			LocalVariableMap vars = new LocalVariableMap();
			collectReadVariables(pb.getStatementBlock().get_hops(), vars);
			ExecutionContext ec = ExecutionContextFactory.createContext(false, null);
			ec.setVariables(vars);
			val = CostEstimationWrapper.getTimeEstimate(pb, ec, false);	
		}
		else{
			//we need to cost the entire program in order to take in-memory status into account
			ExecutionContext ec = ExecutionContextFactory.createContext();
			val = CostEstimationWrapper.getTimeEstimate(pb.getProgram(), ec);
		}
		
		_cntCostPB ++;
		return val;
	}

	private static double getProgramCosts( Program prog ) 
		throws DMLRuntimeException
	{
		//we need to cost the entire program in order to take in-memory status into account
		ExecutionContext ec = ExecutionContextFactory.createContext();
		double val = CostEstimationWrapper.getTimeEstimate(prog, ec);
		_cntCostPB ++;
		
		return val;
	}

	private static void collectReadVariables( ArrayList<Hop> hops, LocalVariableMap vars )
	{
		if( hops!=null ) {
			Hop.resetVisitStatus(hops);
			for( Hop hop : hops )
				collectReadVariables(hop, vars);
		}		
	}

	private static void collectReadVariables( Hop hop, LocalVariableMap vars )
	{
		if( hop == null )
			return;

		//process childs
		for(Hop hi : hop.getInput())
			collectReadVariables( hi, vars );
		
		//investigate hop exec type and known dimensions
		if(    hop instanceof DataOp && hop.getDataType()==DataType.MATRIX
			&& (((DataOp)hop).getDataOpType()==DataOpTypes.TRANSIENTREAD
			||  ((DataOp)hop).getDataOpType()==DataOpTypes.PERSISTENTREAD) ) 
		{
			String varname = hop.getName();
			MatrixCharacteristics mc = new MatrixCharacteristics(hop.getDim1(), hop.getDim2(), 
					    (int)hop.getRowsInBlock(), (int)hop.getColsInBlock(), hop.getNnz());
			MatrixDimensionsMetaData md = new MatrixDimensionsMetaData(mc);
			MatrixObject mo = new MatrixObject(ValueType.DOUBLE, "/tmp", md);
			vars.put(varname, mo);
		}
		
		hop.setVisited();
	}

	private static ArrayList<ProgramBlock> pruneProgramBlocks( ArrayList<ProgramBlock> B ) 
		throws HopsException
	{
		//prune all program blocks w/o mr instructions (mr budget does not matter)
		if( PRUNING_SMALL ){
			ArrayList<ProgramBlock> Bp = new ArrayList<ProgramBlock>();
			for( ProgramBlock pb : B )
				if( OptTreeConverter.containsMRJobInstruction(pb.getInstructions(), false, true) )
					Bp.add( pb );
			B = Bp;
		}
		
		//prune all program blocks, where all mr hops are due to unknowns
		if( PRUNING_UNKNOWN ){
			ArrayList<ProgramBlock> Bp = new ArrayList<ProgramBlock>();
			for( ProgramBlock pb : B )
				if( !pruneHasOnlyUnknownMR(pb) )
					Bp.add( pb );
			B = Bp;
		}
		
		return B;		
	}

	private static boolean pruneHasOnlyUnknownMR( ProgramBlock pb ) 
		throws HopsException
	{
		if (pb instanceof WhileProgramBlock)
		{
			WhileStatementBlock sb = (WhileStatementBlock) pb.getStatementBlock();
			sb.getPredicateHops().resetVisitStatus();
			return pruneHasOnlyUnknownMR(sb.getPredicateHops());
		}	
		else if (pb instanceof IfProgramBlock)
		{
			IfStatementBlock sb = (IfStatementBlock) pb.getStatementBlock();
			sb.getPredicateHops().resetVisitStatus();
			return pruneHasOnlyUnknownMR(sb.getPredicateHops());
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			ForStatementBlock sb = (ForStatementBlock) pb.getStatementBlock();
			sb.getFromHops().resetVisitStatus();
			sb.getToHops().resetVisitStatus();
			sb.getIncrementHops().resetVisitStatus();
			return    pruneHasOnlyUnknownMR(sb.getFromHops())
				   && pruneHasOnlyUnknownMR(sb.getToHops())
				   && pruneHasOnlyUnknownMR(sb.getIncrementHops());
		}
		else //last-level program blocks
		{
			StatementBlock sb = pb.getStatementBlock();
			return pruneHasOnlyUnknownMR(sb.get_hops());
		}
	}

	private static boolean pruneHasOnlyUnknownMR( ArrayList<Hop> hops ) 
		throws HopsException
	{
		boolean ret = false;

		if( hops!=null ){
			ret = true;
			Hop.resetVisitStatus(hops);
			for( Hop hop : hops )
				ret &= pruneHasOnlyUnknownMR(hop);
		}
		
		return ret;
	}

	private static boolean pruneHasOnlyUnknownMR( Hop hop )
	{
		if( hop == null || hop.isVisited() )
			return true;

		boolean ret = true;
		
		//process childs
		for(Hop hi : hop.getInput())
			ret &= pruneHasOnlyUnknownMR( hi );
		
		//investigate hop exec type and known dimensions
		if( hop.getExecType()==ExecType.MR ) {
			boolean lret = false;
			
			//1) operator output dimensions unknown
			lret |= !hop.dimsKnown(); 
				
			//2) operator output dimensions known but inputs unknown
			//(use cases for e.g. AggUnary with scalar output, Binary with one known input)
			for(Hop hi : hop.getInput())
				lret |= !hi.dimsKnown();
				
			ret &= lret;
		}
		
		hop.setVisited();
		
		return ret;
	}

	private static ArrayList<Long> enumerateGridPoints( ArrayList<ProgramBlock> prog, long min, long max, GridEnumType type ) 
		throws DMLRuntimeException, HopsException
	{
		//create enumerator
		GridEnumeration ge = null;
		switch( type ){
			case EQUI_GRID:
				ge = new GridEnumerationEqui(prog, min, max); break;
			case EXP_GRID:
				ge = new GridEnumerationExp(prog, min, max); break;
			case MEM_EQUI_GRID:
				ge = new GridEnumerationMemory(prog, min, max); break;
			case HYBRID_MEM_EXP_GRID:
				ge = new GridEnumerationHybrid(prog, min, max); break;
			default:
				throw new DMLRuntimeException("Unsupported grid enumeration type: "+type);
		}
		
		//generate points 
		ArrayList<Long> ret = ge.enumerateGridPoints();
		LOG.debug("Gen: min="+YarnOptimizerUtils.toMB(min)+", max="+YarnOptimizerUtils.toMB(max)+", npoints="+ret.size());
		
		return ret;
	}

	private static double[][] initLocalMemoTable( ArrayList<ProgramBlock> Bp, double min ) 
		throws DMLRuntimeException
	{
		//allocate memo structure
		int len = Bp.size();
		double[][] memo = new double[len][2];
		
		//init with min resource and current costs
		for( int i=0; i<len; i++ )
		{
			ProgramBlock pb = Bp.get(i);
			ExecutionContext ec = ExecutionContextFactory.createContext();
			memo[i][0] = min;
			memo[i][1] = CostEstimationWrapper.getTimeEstimate(pb.getProgram(), ec);
		}
		
		return memo;
	}

	private static double[][] initGlobalMemoTable( ArrayList<ProgramBlock> B, ArrayList<ProgramBlock> Bp, double[][] lmemo, double min )
	{
		//allocate memo structure
		int len = B.size();
		int lenp = Bp.size(); //lenp<=len
		double[][] memo = new double[len][2];
		
		//init with min resources
		for( int i=0; i<len; i++ ) {
			memo[i][0] = min;
			memo[i][1] = -1;
		}
		
		//overwrite existing values
		int j = 0;
		for( int i=0; i<len && j<lenp; i++ )
		{
			ProgramBlock pb = B.get(i);
			if( pb != Bp.get(j) )
				continue; 
			
			//map local memo entry
			memo[i][0] = lmemo[j][0];
			memo[i][1] = -1;
			j++;
		}
		
		return memo;
	}

	public static void initStatistics()
	{
		_cntCompilePB = 0;
		_cntCostPB = 0;
	}
	
	
	////////
	// old code
	
	public static long jvmToPhy(long jvm, boolean mrRealRun) {
		long ret = (long) Math.ceil((double)jvm * DMLYarnClient.MEM_FACTOR);
		if (mrRealRun) {
			long lowerBound = (long)YarnClusterAnalyzer.getMinMRContarinerPhyMB() * 1024 * 1024;
			if (ret < lowerBound)
				return lowerBound;
		}
		return ret;
	}
	
	public static long budgetToJvm(double budget) {
		return (long) Math.ceil(budget / OptimizerUtils.MEM_UTIL_FACTOR);
	}
	

	public static double phyToBudget(long physical) throws IOException {
		return (double)physical / DMLYarnClient.MEM_FACTOR * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
}
