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

package org.apache.sysml.hops.ipa;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.FunctionOp.FunctionType;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.ExternalFunctionStatement;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.udf.lib.DynamicReadMatrixCP;
import org.apache.sysml.udf.lib.DynamicReadMatrixRcCP;
import org.apache.sysml.udf.lib.OrderWrapper;

/**
 * This Inter Procedural Analysis (IPA) serves two major purposes:
 *   1) Inter-Procedure Analysis: propagate statistics from calling program into 
 *      functions and back into main program. This is done recursively for nested 
 *      function invocations.
 *   2) Intra-Procedural Analysis: propagate statistics across hop dags of subsequent 
 *      statement blocks in order to allow chained function calls and reasoning about
 *      changing sparsity etc (that requires the rewritten hops dag as input). This 
 *      also includes control-flow aware propagation of size and sparsity. Furthermore,
 *      it also serves as a second constant propagation pass.
 *   
 *  Additionally, IPA also covers the removal of unused functions, the decision on
 *  recompile once functions, the removal of unnecessary checkpoints, and the 
 *  global removal of constant binary operations such as X * ones.
 *         
 */
@SuppressWarnings("deprecation")
public class InterProceduralAnalysis 
{
	private static final boolean LDEBUG = false; //internal local debug level
	private static final Log LOG = LogFactory.getLog(InterProceduralAnalysis.class.getName());
    
	//internal configuration parameters
	protected static final boolean INTRA_PROCEDURAL_ANALYSIS      = true; //propagate statistics across statement blocks (main/functions)	
	protected static final boolean PROPAGATE_KNOWN_UDF_STATISTICS = true; //propagate statistics for known external functions 
	protected static final boolean ALLOW_MULTIPLE_FUNCTION_CALLS  = true; //propagate consistent statistics from multiple calls 
	protected static final boolean REMOVE_UNUSED_FUNCTIONS        = true; //remove unused functions (inlined or never called)
	protected static final boolean FLAG_FUNCTION_RECOMPILE_ONCE   = true; //flag functions which require recompilation inside a loop for full function recompile
	protected static final boolean REMOVE_UNNECESSARY_CHECKPOINTS = true; //remove unnecessary checkpoints (unconditionally overwritten intermediates) 
	protected static final boolean REMOVE_CONSTANT_BINARY_OPS     = true; //remove constant binary operations (e.g., X*ones, where ones=matrix(1,...)) 
	protected static final boolean PROPAGATE_SCALAR_VARS_INTO_FUN = true; //propagate scalar variables into functions that are called once
	public static boolean UNARY_DIMS_PRESERVING_FUNS = true; //determine and exploit unary dimension preserving functions 
	
	static {
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("org.apache.sysml.hops.ipa.InterProceduralAnalysis")
				  .setLevel((Level) Level.DEBUG);
		}
	}
	
	private final DMLProgram _prog;
	private final StatementBlock _sb;
	
	//function call graph for functions reachable from main
	private final FunctionCallGraph _fgraph;
	
	//set IPA passes to apply in order 
	private final ArrayList<IPAPass> _passes;
	
	/**
	 * Creates a handle for performing inter-procedural analysis
	 * for a given DML program and its associated HOP DAGs. This
	 * call initializes various internal information such as the
	 * function call graph and auxiliary function call information
	 * which can be reused across multiple IPA calls (e.g., for 
	 * second chance analysis).
	 * 
	 */
	public InterProceduralAnalysis(DMLProgram dmlp) {
		//analyzes the function call graph 
		_prog = dmlp;
		_sb = null;
		_fgraph = new FunctionCallGraph(dmlp);
		
		//create order list of IPA passes
		_passes = new ArrayList<IPAPass>();
		_passes.add(new IPAPassRemoveUnusedFunctions());
		_passes.add(new IPAPassFlagFunctionsRecompileOnce());
		_passes.add(new IPAPassRemoveUnnecessaryCheckpoints());
		_passes.add(new IPAPassRemoveConstantBinaryOps());
	}
	
	public InterProceduralAnalysis(StatementBlock sb) {
		//analyzes the function call graph 
		_prog = sb.getDMLProg();
		_sb = sb;
		_fgraph = new FunctionCallGraph(sb);
		
		//create order list of IPA passes
		_passes = new ArrayList<IPAPass>();
	}
	
	/**
	 * Public interface to perform IPA over a given DML program.
	 * 
	 * @param dmlp the dml program
	 * @throws HopsException if HopsException occurs
	 * @throws ParseException if ParseException occurs
	 * @throws LanguageException if LanguageException occurs
	 */
	public void analyzeProgram() 
		throws HopsException, ParseException, LanguageException
	{
		//TODO move main IPA into separate IPA pass for size propagation
		//together with rework of candidate selection
		
		//step 1: get candidates for statistics propagation into functions (if required)
		Map<String, Integer> fcandCounts = new HashMap<String, Integer>();
		Map<String, FunctionOp> fcandHops = new HashMap<String, FunctionOp>();
		Map<String, Set<Long>> fcandSafeNNZ = new HashMap<String, Set<Long>>(); 
		if( !_prog.getFunctionStatementBlocks().isEmpty() ) {
			for ( StatementBlock sb : _prog.getStatementBlocks() ) //get candidates (over entire program)
				getFunctionCandidatesForStatisticPropagation( sb, fcandCounts, fcandHops );
			pruneFunctionCandidatesForStatisticPropagation( fcandCounts, fcandHops );	
			determineFunctionCandidatesNNZPropagation( fcandHops, fcandSafeNNZ );
			DMLTranslator.resetHopsDAGVisitStatus( _prog );
		}
		
		//step 2: get unary dimension-preserving non-candidate functions
		Collection<String> unaryFcandTmp = _fgraph.getReachableFunctions(fcandCounts.keySet());
		HashSet<String> unaryFcands = new HashSet<String>();
		if( !unaryFcandTmp.isEmpty() && UNARY_DIMS_PRESERVING_FUNS ) {
			for( String tmp : unaryFcandTmp )
				if( isUnarySizePreservingFunction(_prog.getFunctionStatementBlock(tmp)) )
					unaryFcands.add(tmp);
		}
		
		//step 3: propagate statistics and scalars into functions and across DAGs
		if( !fcandCounts.isEmpty() || INTRA_PROCEDURAL_ANALYSIS ) {
			//(callVars used to chain outputs/inputs of multiple functions calls) 
			LocalVariableMap callVars = new LocalVariableMap();
			for ( StatementBlock sb : _prog.getStatementBlocks() ) //propagate stats into candidates
				propagateStatisticsAcrossBlock( sb, fcandCounts, callVars, fcandSafeNNZ, unaryFcands, new HashSet<String>() );
		}
		
		//step 4: apply additional IPA passes
		for( IPAPass pass : _passes )
			if( pass.isApplicable() )
				pass.rewriteProgram(_prog, _fgraph);
	}
	
	public Set<String> analyzeSubProgram() 
		throws HopsException, ParseException
	{
		DMLTranslator.resetHopsDAGVisitStatus(_sb);
		
		//step 1: get candidates for statistics propagation into functions (if required)
		Map<String, Integer> fcandCounts = new HashMap<String, Integer>();
		Map<String, FunctionOp> fcandHops = new HashMap<String, FunctionOp>();
		Map<String, Set<Long>> fcandSafeNNZ = new HashMap<String, Set<Long>>(); 
		Set<String> allFCandKeys = new HashSet<String>();
		getFunctionCandidatesForStatisticPropagation( _sb, fcandCounts, fcandHops );
		allFCandKeys.addAll(fcandCounts.keySet()); //cp before pruning
		pruneFunctionCandidatesForStatisticPropagation( fcandCounts, fcandHops );	
		determineFunctionCandidatesNNZPropagation( fcandHops, fcandSafeNNZ );
		DMLTranslator.resetHopsDAGVisitStatus( _sb );
		
		if( !fcandCounts.isEmpty() ) {
			//step 2: propagate statistics into functions and across DAGs
			//(callVars used to chain outputs/inputs of multiple functions calls) 
			LocalVariableMap callVars = new LocalVariableMap();
			propagateStatisticsAcrossBlock( _sb, fcandCounts, callVars, fcandSafeNNZ, new HashSet<String>(), new HashSet<String>() );
		}
		
		return fcandCounts.keySet();
	}
	
	
	/////////////////////////////
	// GET FUNCTION CANDIDATES
	//////
	
	private void getFunctionCandidatesForStatisticPropagation( StatementBlock sb, Map<String, Integer> fcandCounts, Map<String, FunctionOp> fcandHops ) 
		throws HopsException, ParseException
	{
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sbi : fstmt.getBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcandCounts, fcandHops);
		}
		else if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			for (StatementBlock sbi : wstmt.getBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcandCounts, fcandHops);
		}	
		else if (sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			for (StatementBlock sbi : istmt.getIfBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcandCounts, fcandHops);
			for (StatementBlock sbi : istmt.getElseBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcandCounts, fcandHops);
		}
		else if (sb instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			for (StatementBlock sbi : fstmt.getBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcandCounts, fcandHops);
		}
		else //generic (last-level)
		{
			ArrayList<Hop> roots = sb.get_hops();
			if( roots != null ) //empty statement blocks
				for( Hop root : roots )
					getFunctionCandidatesForStatisticPropagation(sb.getDMLProg(), root, fcandCounts, fcandHops);
		}
	}
	
	private void getFunctionCandidatesForStatisticPropagation(DMLProgram prog, Hop hop, Map<String, Integer> fcandCounts, Map<String, FunctionOp> fcandHops ) 
		throws HopsException, ParseException
	{
		if( hop.isVisited() )
			return;
		
		if( hop instanceof FunctionOp && !((FunctionOp)hop).getFunctionNamespace().equals(DMLProgram.INTERNAL_NAMESPACE) )
		{
			//maintain counters and investigate functions if not seen so far
			FunctionOp fop = (FunctionOp) hop;
			String fkey = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(), fop.getFunctionName());
			
			if( fcandCounts.containsKey(fkey) ) {
				if( ALLOW_MULTIPLE_FUNCTION_CALLS )
				{
					//compare input matrix characteristics for both function calls
					//(if unknown or difference: maintain counter - this function is no candidate)
					boolean consistent = true;
					FunctionOp efop = fcandHops.get(fkey);
					int numInputs = efop.getInput().size();
					for( int i=0; i<numInputs; i++ )
					{
						Hop h1 = efop.getInput().get(i);
						Hop h2 = fop.getInput().get(i);
						//check matrix and scalar sizes (if known dims, nnz known/unknown, 
						// safeness of nnz propagation, determined later per input)
						consistent &= (h1.dimsKnown() && h2.dimsKnown()
								   &&  h1.getDim1()==h2.getDim1() 
								   &&  h1.getDim2()==h2.getDim2()
								   &&  h1.getNnz()==h2.getNnz() );
						//check literal values (equi value)
						if( h1 instanceof LiteralOp ){
							consistent &= (h2 instanceof LiteralOp 
									      && HopRewriteUtils.isEqualValue((LiteralOp)h1, (LiteralOp)h2));
						}
						
						
					}
					
					if( !consistent ) //if differences, do not propagate
						fcandCounts.put(fkey, fcandCounts.get(fkey)+1);
				}
				else
				{
					//maintain counter (this function is no candidate)
					fcandCounts.put(fkey, fcandCounts.get(fkey)+1);
				}
			}
			else { //first appearance
				fcandCounts.put(fkey, 1); //create a new count entry
				fcandHops.put(fkey, fop); //keep the function call hop
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
				getFunctionCandidatesForStatisticPropagation(fsb, fcandCounts, fcandHops);
			}
		}
			
		for( Hop c : hop.getInput() )
			getFunctionCandidatesForStatisticPropagation(prog, c, fcandCounts, fcandHops);
		
		hop.setVisited();
	}
	
	private void pruneFunctionCandidatesForStatisticPropagation(Map<String, Integer> fcandCounts, Map<String, FunctionOp> fcandHops)
	{
		//debug input
		if( LOG.isDebugEnabled() )
			for( Entry<String,Integer> e : fcandCounts.entrySet() )
			{
				String key = e.getKey();
				Integer count = e.getValue();
				LOG.debug("IPA: FUNC statistic propagation candidate: "+key+", callCount="+count);
			}
		
		//materialize key set
		Set<String> tmp = new HashSet<String>(fcandCounts.keySet());
		
		//check and prune candidate list
		for( String key : tmp )
		{
			Integer cnt = fcandCounts.get(key);
			if( cnt != null && cnt > 1 ) //if multiple refs
				fcandCounts.remove(key);
		}
		
		//debug output
		if( LOG.isDebugEnabled() )
			for( String key : fcandCounts.keySet() )
			{
				LOG.debug("IPA: FUNC statistic propagation candidate (after pruning): "+key);
			}
	}
	
	private boolean isUnarySizePreservingFunction(FunctionStatementBlock fsb) 
		throws HopsException, ParseException 
	{
		FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
		
		//check unary functions over matrices
		boolean ret = (fstmt.getInputParams().size() == 1 
				&& fstmt.getInputParams().get(0).getDataType()==DataType.MATRIX
				&& fstmt.getOutputParams().size() == 1
				&& fstmt.getOutputParams().get(0).getDataType()==DataType.MATRIX);
		
		//check size-preserving characteristic
		if( ret ) {
			HashMap<String, Integer> tmp1 = new HashMap<String,Integer>();
			HashMap<String, Set<Long>> tmp2 = new HashMap<String, Set<Long>>();
			HashSet<String> tmp3 = new HashSet<String>();
			HashSet<String> tmp4 = new HashSet<String>();
			LocalVariableMap callVars = new LocalVariableMap();
			
			//populate input
			MatrixObject mo = createOutputMatrix(7777, 3333, -1);
			callVars.put(fstmt.getInputParams().get(0).getName(), mo);
			
			//propagate statistics
			for (StatementBlock sbi : fstmt.getBody())
				propagateStatisticsAcrossBlock(sbi,  tmp1, callVars, tmp2, tmp3, tmp4);
		
			//compare output
			MatrixObject mo2 = (MatrixObject)callVars.get(fstmt.getOutputParams().get(0).getName());
			ret &= mo.getNumRows() == mo2.getNumRows() && mo.getNumColumns() == mo2.getNumColumns();
		
			//reset function
			mo.getMatrixCharacteristics().setDimension(-1, -1);
			for (StatementBlock sbi : fstmt.getBody())
				propagateStatisticsAcrossBlock(sbi,  tmp1, callVars, tmp2, tmp3, tmp4);
		}
		
		return ret;
	}

	/////////////////////////////
	// DETERMINE NNZ PROPAGATE SAFENESS
	//////

	/**
	 * Populates fcandSafeNNZ with all <functionKey,hopID> pairs where it is safe to
	 * propagate nnz into the function.
	 *  
	 * @param fcandHops function candidate HOPs
	 * @param fcandSafeNNZ function candidate safe non-zeros
	 */
	private void determineFunctionCandidatesNNZPropagation(Map<String, FunctionOp> fcandHops, Map<String, Set<Long>> fcandSafeNNZ)
	{
		//for all function candidates
		for( Entry<String, FunctionOp> e : fcandHops.entrySet() )
		{
			String fKey = e.getKey();
			FunctionOp fop = e.getValue();
			HashSet<Long> tmp = new HashSet<Long>();
			
			//for all inputs of this function call
			for( Hop input : fop.getInput() )
			{
				//if nnz known it is safe to propagate those nnz because for multiple calls 
				//we checked of equivalence and hence all calls have the same nnz
				if( input.getNnz()>=0 ) 
					tmp.add(input.getHopID());
			}
			
			fcandSafeNNZ.put(fKey, tmp);
		}
	}
	
	/////////////////////////////
	// INTRA-PROCEDURE ANALYSIS
	//////	

	/**
	 * Perform intra-procedural analysis (IPA) by propagating statistics
	 * across statement blocks.
	 *
	 * @param sb  DML statement blocks.
	 * @param fcand  Function candidates.
	 * @param callVars  Map of variables eligible for propagation.
	 * @param fcandSafeNNZ  Function candidate safe non-zeros.
	 * @param unaryFcands  Unary function candidates.
	 * @param fnStack  Function stack to determine current scope.
	 * @throws HopsException  If a HopsException occurs.
	 * @throws ParseException  If a ParseException occurs.
	 */
	private void propagateStatisticsAcrossBlock( StatementBlock sb, Map<String, Integer> fcand, LocalVariableMap callVars, Map<String, Set<Long>> fcandSafeNNZ, Set<String> unaryFcands, Set<String> fnStack )
		throws HopsException, ParseException
	{
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sbi : fstmt.getBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
		}
		else if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			//old stats into predicate
			propagateStatisticsAcrossPredicateDAG(wsb.getPredicateHops(), callVars);
			//remove updated constant scalars
			Recompiler.removeUpdatedScalars(callVars, wsb);
			//check and propagate stats into body
			LocalVariableMap oldCallVars = (LocalVariableMap) callVars.clone();
			for (StatementBlock sbi : wstmt.getBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
			if( Recompiler.reconcileUpdatedCallVarsLoops(oldCallVars, callVars, wsb) ){ //second pass if required
				propagateStatisticsAcrossPredicateDAG(wsb.getPredicateHops(), callVars);
				for (StatementBlock sbi : wstmt.getBody())
					propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
			}
			//remove updated constant scalars
			Recompiler.removeUpdatedScalars(callVars, sb);
		}	
		else if (sb instanceof IfStatementBlock) 
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			//old stats into predicate
			propagateStatisticsAcrossPredicateDAG(isb.getPredicateHops(), callVars);			
			//check and propagate stats into body
			LocalVariableMap oldCallVars = (LocalVariableMap) callVars.clone();
			LocalVariableMap callVarsElse = (LocalVariableMap) callVars.clone();
			for (StatementBlock sbi : istmt.getIfBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
			for (StatementBlock sbi : istmt.getElseBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVarsElse, fcandSafeNNZ, unaryFcands, fnStack);
			callVars = Recompiler.reconcileUpdatedCallVarsIf(oldCallVars, callVars, callVarsElse, isb);
			//remove updated constant scalars
			Recompiler.removeUpdatedScalars(callVars, sb);
		}
		else if (sb instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			//old stats into predicate
			propagateStatisticsAcrossPredicateDAG(fsb.getFromHops(), callVars);
			propagateStatisticsAcrossPredicateDAG(fsb.getToHops(), callVars);
			propagateStatisticsAcrossPredicateDAG(fsb.getIncrementHops(), callVars);
			//remove updated constant scalars
			Recompiler.removeUpdatedScalars(callVars, fsb);
			//check and propagate stats into body
			LocalVariableMap oldCallVars = (LocalVariableMap) callVars.clone();
			for (StatementBlock sbi : fstmt.getBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
			if( Recompiler.reconcileUpdatedCallVarsLoops(oldCallVars, callVars, fsb) )
				for (StatementBlock sbi : fstmt.getBody())
					propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
			//remove updated constant scalars
			Recompiler.removeUpdatedScalars(callVars, sb);
		}
		else //generic (last-level)
		{	
			//remove updated constant scalars
			Recompiler.removeUpdatedScalars(callVars, sb);
			//old stats in, new stats out if updated
			ArrayList<Hop> roots = sb.get_hops();
			DMLProgram prog = sb.getDMLProg();
			//replace scalar reads with literals
			Hop.resetVisitStatus(roots);
			propagateScalarsAcrossDAG(roots, callVars);
			//refresh stats across dag
			Hop.resetVisitStatus(roots);
			propagateStatisticsAcrossDAG(roots, callVars);
			//propagate stats into function calls
			Hop.resetVisitStatus(roots);
			propagateStatisticsIntoFunctions(prog, roots, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
		}
	}

	/**
	 * Propagate scalar values across DAGs.
	 *
	 * This replaces scalar reads and typecasts thereof with literals.
	 *
	 * Ultimately, this leads to improvements because the size
	 * expression evaluation over DAGs with scalar symbol table entries
	 * (which is also applied during IPA) is limited to supported
	 * operations, whereas literal replacement is a brute force method
	 * that applies to all (including future) operations.
	 *
	 * @param roots  List of HOPs.
	 * @param vars  Map of variables eligible for propagation.
	 * @throws HopsException  If a HopsException occurs.
	 */
	private void propagateScalarsAcrossDAG(ArrayList<Hop> roots, LocalVariableMap vars)
		throws HopsException
	{
		for (Hop hop : roots) {
			try {
				Recompiler.rReplaceLiterals(hop, vars, true);
			} catch (Exception ex) {
				throw new HopsException("Failed to perform scalar literal replacement.", ex);
			}
		}
	}

	private void propagateStatisticsAcrossPredicateDAG( Hop root, LocalVariableMap vars ) 
		throws HopsException
	{
		if( root == null )
			return;
		
		//reset visit status because potentially called multiple times
		root.resetVisitStatus();
		
		try
		{
			Recompiler.rUpdateStatistics( root, vars );
			
			//note: for predicates no output statistics
			//Recompiler.extractDAGOutputStatistics(root, vars);
		}
		catch(Exception ex)
		{
			throw new HopsException("Failed to update Hop DAG statistics.", ex);
		}
	}

	/**
	 * Propagate matrix sizes across DAGs.
	 *
	 * @param roots  List of HOP DAG root nodes.
	 * @param vars  Map of variables eligible for propagation.
	 * @throws HopsException  If a HopsException occurs.
	 */
	private void propagateStatisticsAcrossDAG( ArrayList<Hop> roots, LocalVariableMap vars )
		throws HopsException
	{
		if( roots == null )
			return;
		
		try
		{
			//update DAG statistics from leafs to roots
			for( Hop hop : roots )
				Recompiler.rUpdateStatistics( hop, vars );

			//extract statistics from roots
			Recompiler.extractDAGOutputStatistics(roots, vars, true);
		}
		catch( Exception ex )
		{
			throw new HopsException("Failed to update Hop DAG statistics.", ex);
		}
	}
	
	
	/////////////////////////////
	// INTER-PROCEDURE ANALYIS
	//////

	/**
	 * Propagate statistics from the calling program into a function
	 * block.
	 *
	 * @param prog  The DML program.
	 * @param roots List of HOP DAG root notes for propagation.
	 * @param fcand  Function candidates.
	 * @param callVars  Calling program's map of variables eligible for
	 *                     propagation.
	 * @param fcandSafeNNZ  Function candidate safe non-zeros.
	 * @param unaryFcands  Unary function candidates.
	 * @param fnStack  Function stack to determine current scope.
	 * @throws HopsException  If a HopsException occurs.
	 * @throws ParseException  If a ParseException occurs.
	 */
	private void propagateStatisticsIntoFunctions(DMLProgram prog, ArrayList<Hop> roots, Map<String, Integer> fcand, LocalVariableMap callVars, Map<String, Set<Long>> fcandSafeNNZ, Set<String> unaryFcands, Set<String> fnStack )
			throws HopsException, ParseException
	{
		for( Hop root : roots )
			propagateStatisticsIntoFunctions(prog, root, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
	}

	/**
	 * Propagate statistics from the calling program into a function
	 * block.
	 *
	 * @param prog  The DML program.
	 * @param hop HOP to propagate statistics into.
	 * @param fcand  Function candidates.
	 * @param callVars  Calling program's map of variables eligible for
	 *                     propagation.
	 * @param fcandSafeNNZ  Function candidate safe non-zeros.
	 * @param unaryFcands  Unary function candidates.
	 * @param fnStack  Function stack to determine current scope.
	 * @throws HopsException  If a HopsException occurs.
	 * @throws ParseException  If a ParseException occurs.
	 */
	private void propagateStatisticsIntoFunctions(DMLProgram prog, Hop hop, Map<String, Integer> fcand, LocalVariableMap callVars, Map<String, Set<Long>> fcandSafeNNZ, Set<String> unaryFcands, Set<String> fnStack ) 
		throws HopsException, ParseException
	{
		if( hop.isVisited() )
			return;
		
		for( Hop c : hop.getInput() )
			propagateStatisticsIntoFunctions(prog, c, fcand, callVars, fcandSafeNNZ, unaryFcands, fnStack);
		
		if( hop instanceof FunctionOp )
		{
			//maintain counters and investigate functions if not seen so far
			FunctionOp fop = (FunctionOp) hop;
			String fkey = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(), fop.getFunctionName());
			
			if( fop.getFunctionType() == FunctionType.DML )
			{
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				
				if(  fcand.containsKey(fkey) && 
				    !fnStack.contains(fkey)  ) //prevent recursion	
				{
					//maintain function call stack
					fnStack.add(fkey);
					
					//create mapping and populate symbol table for refresh
					LocalVariableMap tmpVars = new LocalVariableMap();
					populateLocalVariableMapForFunctionCall( fstmt, fop, 
							callVars, tmpVars, fcandSafeNNZ.get(fkey), fcand.get(fkey) );
	
					//recursively propagate statistics
					propagateStatisticsAcrossBlock(fsb, fcand, tmpVars, fcandSafeNNZ, unaryFcands, fnStack);
					
					//extract vars from symbol table, re-map and refresh main program
					extractFunctionCallReturnStatistics(fstmt, fop, tmpVars, callVars, true);		
					
					//maintain function call stack
					fnStack.remove(fkey);
				}
				else if( unaryFcands.contains(fkey) ) {
					extractFunctionCallEquivalentReturnStatistics(fstmt, fop, callVars);
				}
				else {
					extractFunctionCallUnknownReturnStatistics(fstmt, fop, callVars);
				}
			}
			else if (   fop.getFunctionType() == FunctionType.EXTERNAL_FILE
				     || fop.getFunctionType() == FunctionType.EXTERNAL_MEM  )
			{
				//infer output size for known external functions
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
				ExternalFunctionStatement fstmt = (ExternalFunctionStatement) fsb.getStatement(0);
				if( PROPAGATE_KNOWN_UDF_STATISTICS ) 
					extractExternalFunctionCallReturnStatistics(fstmt, fop, callVars);
				else
					extractFunctionCallUnknownReturnStatistics(fstmt, fop, callVars);
			}
		}
		
		hop.setVisited();
	}
	
	private void populateLocalVariableMapForFunctionCall( FunctionStatement fstmt, FunctionOp fop, LocalVariableMap callvars, LocalVariableMap vars, Set<Long> inputSafeNNZ, Integer numCalls ) 
		throws HopsException
	{
		ArrayList<DataIdentifier> inputVars = fstmt.getInputParams();
		ArrayList<Hop> inputOps = fop.getInput();
		
		for( int i=0; i<inputVars.size(); i++ )
		{
			//create mapping between input hops and vars
			DataIdentifier dat = inputVars.get(i);
			Hop input = inputOps.get(i);
			
			if( input.getDataType()==DataType.MATRIX )
			{
				//propagate matrix characteristics
				MatrixObject mo = new MatrixObject(ValueType.DOUBLE, null);
				MatrixCharacteristics mc = new MatrixCharacteristics( 
											input.getDim1(), input.getDim2(), 
											ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize(),
											inputSafeNNZ.contains(input.getHopID())?input.getNnz():-1 );
				MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,null,null);
				mo.setMetaData(meta);	
				vars.put(dat.getName(), mo);	
			}
			else if( input.getDataType()==DataType.SCALAR )
			{
				//always propagate scalar literals into functions
				//(for multiple calls, literal equivalence already checked)
				if( input instanceof LiteralOp ) {
					vars.put(dat.getName(), ScalarObjectFactory
						.createScalarObject(input.getValueType(), (LiteralOp)input));
				}
				//propagate scalar variables into functions if called once
				//and input scalar is existing variable in symbol table
				else if( PROPAGATE_SCALAR_VARS_INTO_FUN 
					&& numCalls != null && numCalls == 1
					&& input instanceof DataOp  ) 
				{
					Data scalar = callvars.get(input.getName()); 
					if( scalar != null && scalar instanceof ScalarObject ) {
						vars.put(dat.getName(), scalar);
					}
				}
			}
		}
	}

	/**
	 * Extract return variable statistics from this function into the
	 * calling program.
	 *
	 * @param fstmt  The function statement.
	 * @param fop  The function op.
	 * @param tmpVars  Function's map of variables eligible for
	 *                    extraction.
	 * @param callVars  Calling program's map of variables.
	 * @param overwrite  Whether or not to overwrite variables in the
	 *                      calling program's variable map.
	 * @throws HopsException  If a HopsException occurs.
	 */
	private void extractFunctionCallReturnStatistics( FunctionStatement fstmt, FunctionOp fop, LocalVariableMap tmpVars, LocalVariableMap callVars, boolean overwrite ) 
		throws HopsException
	{
		ArrayList<DataIdentifier> foutputOps = fstmt.getOutputParams();
		String[] outputVars = fop.getOutputVariableNames();
		String fkey = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(), fop.getFunctionName());
		
		try
		{
			for( int i=0; i<foutputOps.size(); i++ )
			{
				DataIdentifier di = foutputOps.get(i);
				String fvarname = di.getName(); //name in function signature
				String pvarname = outputVars[i]; //name in calling program

				// If the calling program is reassigning a variable with the output of this
				// function, and the datatype differs between that variable and this function
				// output, remove that variable from the calling program's variable map.
				if( callVars.keySet().contains(pvarname) ) {
					DataType fdataType = di.getDataType();
					DataType pdataType = callVars.get(pvarname).getDataType();
					if( fdataType != pdataType ) {
						// datatype has changed, and the calling program is reassigning the
						// the variable, so remove it from the calling variable map
						callVars.remove(pvarname);
					}
				}
				// Update or add to the calling program's variable map.
				if( di.getDataType()==DataType.MATRIX && tmpVars.keySet().contains(fvarname) )
				{
					MatrixObject moIn = (MatrixObject) tmpVars.get(fvarname);
					
					if( !callVars.keySet().contains(pvarname) || overwrite ) //not existing so far
					{
						MatrixObject moOut = createOutputMatrix(moIn.getNumRows(), moIn.getNumColumns(), moIn.getNnz());	
						callVars.put(pvarname, moOut);
					}
					else //already existing: take largest   
					{
						Data dat = callVars.get(pvarname);
						if( dat instanceof MatrixObject )
						{
							MatrixObject moOut = (MatrixObject)dat;
							MatrixCharacteristics mc = moOut.getMatrixCharacteristics();
							if( OptimizerUtils.estimateSizeExactSparsity(mc.getRows(), mc.getCols(), (mc.getNonZeros()>0)?((double)mc.getNonZeros())/mc.getRows()/mc.getCols():1.0)	
							    < OptimizerUtils.estimateSize(moIn.getNumRows(), moIn.getNumColumns()) )
							{
								//update statistics if necessary
								mc.setDimension(moIn.getNumRows(), moIn.getNumColumns());
								mc.setNonZeros(moIn.getNnz());
							}
						}
						
					}
				}
			}
		}
		catch( Exception ex )
		{
			throw new HopsException( "Failed to extract output statistics of function "+fkey+".", ex);
		}
	}
	
	private void extractFunctionCallUnknownReturnStatistics( FunctionStatement fstmt, FunctionOp fop, LocalVariableMap callVars ) 
		throws HopsException
	{
		ArrayList<DataIdentifier> foutputOps = fstmt.getOutputParams();
		String[] outputVars = fop.getOutputVariableNames();
		String fkey = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(), fop.getFunctionName());
		
		try
		{
			for( int i=0; i<foutputOps.size(); i++ )
			{
				DataIdentifier di = foutputOps.get(i);
				String pvarname = outputVars[i]; //name in calling program
				
				if( di.getDataType()==DataType.MATRIX )
				{
					MatrixObject moOut = createOutputMatrix(-1, -1, -1);	
					callVars.put(pvarname, moOut);
				}
			}
		}
		catch( Exception ex )
		{
			throw new HopsException( "Failed to extract output statistics of function "+fkey+".", ex);
		}
	}
	
	private void extractFunctionCallEquivalentReturnStatistics( FunctionStatement fstmt, FunctionOp fop, LocalVariableMap callVars ) 
		throws HopsException
	{
		String fkey = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(), fop.getFunctionName());
		try {
			Hop input = fop.getInput().get(0);
			MatrixObject moOut = createOutputMatrix(input.getDim1(), input.getDim2(), -1);	
			callVars.put(fop.getOutputVariableNames()[0], moOut);
		}
		catch( Exception ex ) {
			throw new HopsException( "Failed to extract output statistics for unary function "+fkey+".", ex);
		}
	}
	
	private void extractExternalFunctionCallReturnStatistics( ExternalFunctionStatement fstmt, FunctionOp fop, LocalVariableMap callVars ) 
		throws HopsException
	{
		String className = fstmt.getOtherParams().get(ExternalFunctionStatement.CLASS_NAME);

		if( className.equals(OrderWrapper.class.getName()) )
		{			
			Hop input = fop.getInput().get(0);
			long lnnz = className.equals(OrderWrapper.class.getName()) ? input.getNnz() : -1;
			MatrixObject moOut = createOutputMatrix(input.getDim1(), input.getDim2(),lnnz);
			callVars.put(fop.getOutputVariableNames()[0], moOut);
		}
		else if(   className.equals(DynamicReadMatrixCP.class.getName())
				|| className.equals(DynamicReadMatrixRcCP.class.getName()) ) 
		{
			Hop input1 = fop.getInput().get(1); //rows
			Hop input2 = fop.getInput().get(2); //cols
			if( input1 instanceof LiteralOp && input2 instanceof LiteralOp )
				callVars.put(fop.getOutputVariableNames()[0], createOutputMatrix(((LiteralOp)input1).getLongValue(), 
						                                                         ((LiteralOp)input2).getLongValue(),-1));
		}
		else
		{
			extractFunctionCallUnknownReturnStatistics(fstmt, fop, callVars);
		}
	}
	
	private MatrixObject createOutputMatrix( long dim1, long dim2, long nnz ) {
		MatrixObject moOut = new MatrixObject(ValueType.DOUBLE, null);
		MatrixCharacteristics mc = new MatrixCharacteristics( dim1, dim2,
				ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize(), nnz);
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,null,null);
		moOut.setMetaData(meta);
		
		return moOut;
	}
}
