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

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.ipa.InterProceduralAnalysis;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.hops.rewrite.HopRewriteRule;
import org.apache.sysml.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteConstantFolding;
import org.apache.sysml.hops.rewrite.RewriteRemoveUnnecessaryBranches;
import org.apache.sysml.hops.rewrite.StatementBlockRewriteRule;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ParForStatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.opt.Optimizer.CostModelType;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Stat;
import org.apache.sysml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import org.apache.sysml.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.utils.Statistics;


/**
 * Wrapper to ParFOR cost estimation and optimizer. This is intended to be the 
 * only public access to the optimizer package.
 * 
 * NOTE: There are two main alternatives for invocation of this OptimizationWrapper:
 * (1) During compilation (after creating rtprog), (2) on execute of all top-level ParFOR PBs.
 * We decided to use (2) (and carry the SBs during execution) due to the following advantages
 *   - Known Statistics: problem size of top-level parfor known, in general, less unknown statistics
 *   - No Overhead: preventing overhead for non-parfor scripts (finding top-level parfors)
 *   - Simplicity: no need of finding top-level parfors 
 * 
 */
public class OptimizationWrapper 
{
	
	private static final boolean LDEBUG = false; //internal local debug level
	private static final Log LOG = LogFactory.getLog(OptimizationWrapper.class.getName());
	
	//internal parameters
	public static final double PAR_FACTOR_INFRASTRUCTURE = 1.0;
	private static final boolean CHECK_PLAN_CORRECTNESS = false; 
	
	static
	{
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("org.apache.sysml.runtime.controlprogram.parfor.opt")
				  .setLevel((Level) Level.DEBUG);
		}
	}

	/**
	 * Called once per top-level parfor (during runtime, on parfor execute)
	 * in order to optimize the specific parfor program block.
	 * 
	 * NOTE: this is the default way to invoke parfor optimizers.
	 * 
	 * @param type ?
	 * @param sb parfor statement block
	 * @param pb parfor program block
	 * @param ec execution context
	 * @param monitor ?
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public static void optimize( POptMode type, ParForStatementBlock sb, ParForProgramBlock pb, ExecutionContext ec, boolean monitor ) 
		throws DMLRuntimeException
	{
		Timing time = new Timing(true);
		
		LOG.debug("ParFOR Opt: Running optimization for ParFOR("+pb.getID()+")");
		
		
		//set max contraints if not specified
		int ck = UtilFunctions.toInt( Math.max( InfrastructureAnalyzer.getCkMaxCP(),
						                        InfrastructureAnalyzer.getCkMaxMR() ) * PAR_FACTOR_INFRASTRUCTURE );
		double cm = InfrastructureAnalyzer.getCmMax() * OptimizerUtils.MEM_UTIL_FACTOR; 
		
		//execute optimizer
		optimize( type, ck, cm, sb, pb, ec, monitor );
		
		double timeVal = time.stop();
		LOG.debug("ParFOR Opt: Finished optimization for PARFOR("+pb.getID()+") in "+timeVal+"ms.");
		//System.out.println("ParFOR Opt: Finished optimization for PARFOR("+pb.getID()+") in "+timeVal+"ms.");
		if( monitor )
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_T, timeVal);
	}

	public static void setLogLevel( Level optLogLevel )
	{
		if( !LDEBUG ){ //set log level if not overwritten by internal flag
			Logger.getLogger("org.apache.sysml.runtime.controlprogram.parfor.opt")
			      .setLevel( optLogLevel );
		}
	}

	@SuppressWarnings("unused")
	private static void optimize( POptMode otype, int ck, double cm, ParForStatementBlock sb, ParForProgramBlock pb, ExecutionContext ec, boolean monitor ) 
		throws DMLRuntimeException
	{
		Timing time = new Timing(true);
		
		//maintain statistics
		if( DMLScript.STATISTICS )
			Statistics.incrementParForOptimCount();
		
		//create specified optimizer
		Optimizer opt = createOptimizer( otype );
		CostModelType cmtype = opt.getCostModelType();
		LOG.trace("ParFOR Opt: Created optimizer ("+otype+","+opt.getPlanInputType()+","+opt.getCostModelType());
		
		OptTree tree = null;
		
		//recompile parfor body 
		if( ConfigurationManager.isDynamicRecompilation() )
		{
			ForStatement fs = (ForStatement) sb.getStatement(0);
			
			//debug output before recompilation
			if( LOG.isDebugEnabled() ) 
			{
				try {
					tree = OptTreeConverter.createOptTree(ck, cm, opt.getPlanInputType(), sb, pb, ec); 
					LOG.debug("ParFOR Opt: Input plan (before recompilation):\n" + tree.explain(false));
					OptTreeConverter.clear();
				}
				catch(Exception ex)
				{
					throw new DMLRuntimeException("Unable to create opt tree.", ex);
				}
			}
			
			//constant propagation into parfor body 
			//(input scalars to parfor are guaranteed read only, but need to ensure safe-replace on multiple reopt
			//separate propagation required because recompile in-place without literal replacement)
			try{
				LocalVariableMap constVars = ProgramRecompiler.getReusableScalarVariables(sb.getDMLProg(), sb, ec.getVariables());
				ProgramRecompiler.replaceConstantScalarVariables(sb, constVars);
			}
			catch(Exception ex){
				throw new DMLRuntimeException(ex);
			}
			
			
			//program rewrites (e.g., constant folding, branch removal) according to replaced literals
			try {
				ProgramRewriter rewriter = createProgramRewriterWithRuleSets();
				ProgramRewriteStatus state = new ProgramRewriteStatus();
				rewriter.rewriteStatementBlockHopDAGs( sb, state );
				fs.setBody(rewriter.rewriteStatementBlocks(fs.getBody(), state));
				if( state.getRemovedBranches() ){
					LOG.debug("ParFOR Opt: Removed branches during program rewrites, rebuilding runtime program");
					pb.setChildBlocks(ProgramRecompiler.generatePartitialRuntimeProgram(pb.getProgram(), fs.getBody()));
				}
			}
			catch(Exception ex){
				throw new DMLRuntimeException(ex);
			}
			
			//recompilation of parfor body and called functions (if safe)
			try{
				//core parfor body recompilation (based on symbol table entries)
				//* clone of variables in order to allow for statistics propagation across DAGs
				//(tid=0, because deep copies created after opt)
				LocalVariableMap tmp = (LocalVariableMap) ec.getVariables().clone();
				Recompiler.recompileProgramBlockHierarchy(pb.getChildBlocks(), tmp, 0, true);
				
				//inter-procedural optimization (based on previous recompilation)
				if( pb.hasFunctions() ) {
					InterProceduralAnalysis ipa = new InterProceduralAnalysis(sb);
					Set<String> fcand = ipa.analyzeSubProgram();
					
					if( !fcand.isEmpty() ) {
						//regenerate runtime program of modified functions
						for( String func : fcand )
						{
							String[] funcparts = DMLProgram.splitFunctionKey(func);
							FunctionProgramBlock fpb = pb.getProgram().getFunctionProgramBlock(funcparts[0], funcparts[1]);
							//reset recompilation flags according to recompileOnce because it is only safe if function is recompileOnce 
							//because then recompiled for every execution (otherwise potential issues if func also called outside parfor)
							Recompiler.recompileProgramBlockHierarchy(fpb.getChildBlocks(), new LocalVariableMap(), 0, fpb.isRecompileOnce());
						}		
					}
				}
			}
			catch(Exception ex){
				throw new DMLRuntimeException(ex);
			}
		}
		
		//create opt tree (before optimization)
		try {
			tree = OptTreeConverter.createOptTree(ck, cm, opt.getPlanInputType(), sb, pb, ec); 
			LOG.debug("ParFOR Opt: Input plan (before optimization):\n" + tree.explain(false));
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to create opt tree.", ex);
		}
		
		//create cost estimator
		CostEstimator est = createCostEstimator( cmtype, ec.getVariables() );
		LOG.trace("ParFOR Opt: Created cost estimator ("+cmtype+")");
		
		//core optimize
		opt.optimize( sb, pb, tree, est, ec );
		LOG.debug("ParFOR Opt: Optimized plan (after optimization): \n" + tree.explain(false));
		
		//assert plan correctness
		if( CHECK_PLAN_CORRECTNESS && LOG.isDebugEnabled() )
		{
			try{
				OptTreePlanChecker.checkProgramCorrectness(pb, sb, new HashSet<String>());
				LOG.debug("ParFOR Opt: Checked plan and program correctness.");
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException("Failed to check program correctness.", ex);
			}
		}
		
		long ltime = (long) time.stop();
		LOG.trace("ParFOR Opt: Optimized plan in "+ltime+"ms.");
		if( DMLScript.STATISTICS )
			Statistics.incrementParForOptimTime(ltime);
		
		//cleanup phase
		OptTreeConverter.clear();
		
		//monitor stats
		if( monitor )
		{
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_OPTIMIZER, otype.ordinal());
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_NUMTPLANS, opt.getNumTotalPlans());
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_NUMEPLANS, opt.getNumEvaluatedPlans());
		}
	}

	private static Optimizer createOptimizer( POptMode otype ) 
		throws DMLRuntimeException
	{
		Optimizer opt = null;
		
		switch( otype )
		{
			case HEURISTIC:
				opt = new OptimizerHeuristic();
				break;
			case RULEBASED:
				opt = new OptimizerRuleBased();
				break;	
			case CONSTRAINED:
				opt = new OptimizerConstrained();
				break;	
			default:
				throw new DMLRuntimeException("Undefined optimizer: '"+otype+"'.");
		}
		
		return opt;
	}

	private static CostEstimator createCostEstimator( CostModelType cmtype, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		CostEstimator est = null;
		
		switch( cmtype )
		{
			case STATIC_MEM_METRIC:
				est = new CostEstimatorHops( 
						OptTreeConverter.getAbstractPlanMapping() );
				break;
			case RUNTIME_METRICS:
				est = new CostEstimatorRuntime( 
						OptTreeConverter.getAbstractPlanMapping(), 
						(LocalVariableMap)vars.clone() );
				break;
			default:
				throw new DMLRuntimeException("Undefined cost model type: '"+cmtype+"'.");
		}
		
		return est;
	}

	private static ProgramRewriter createProgramRewriterWithRuleSets()
	{
		//create hop rewrite set
		ArrayList<HopRewriteRule> hRewrites = new ArrayList<HopRewriteRule>();
		hRewrites.add( new RewriteConstantFolding() );
		
		//create statementblock rewrite set
		ArrayList<StatementBlockRewriteRule> sbRewrites = new ArrayList<StatementBlockRewriteRule>();
		sbRewrites.add( new RewriteRemoveUnnecessaryBranches() );
		
		ProgramRewriter rewriter = new ProgramRewriter( hRewrites, sbRewrites );
		
		return rewriter;
	}
}
