/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.FunctionOp.FunctionType;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.rewrite.HopRewriteUtils;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.packagesupport.DeNaNWrapper;
import com.ibm.bi.dml.packagesupport.DeNegInfinityWrapper;
import com.ibm.bi.dml.packagesupport.DynamicReadMatrixCP;
import com.ibm.bi.dml.packagesupport.DynamicReadMatrixRcCP;
//import com.ibm.bi.dml.packagesupport.EigenWrapper;
//import com.ibm.bi.dml.packagesupport.LinearSolverWrapperCP;
import com.ibm.bi.dml.packagesupport.OrderWrapper;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;

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
 * In general, the basic concepts of IPA are as follows and all places that deal with
 * statistic propagation should adhere to that:
 *   * Rule 1: Exact size propagation: Since the dimension information are sometimes used
 *     for specific lops construction (e.g., in append) and rewrites, we cannot propagate worst-case 
 *     estimates but only exact information; otherwise size must be unknown.
 *   * Rule 2: Dimension information and sparsity are handled separately, i.e., if an updated 
 *     variable has changing sparsity but constant dimensions, its dimensions are known but
 *     sparsity unknown.
 * 
 * More specifically, those two rules are currently realized as follows:
 *   * Statistics propagation is applied for DML-bodied functions that are invoked exactly once.
 *     This ensures that we can savely propagate exact information into this function.
 *     If ALLOW_MULTIPLE_FUNCTION_CALLS is enabled we treat multiple calls with the same sizes
 *     as one call and hence, propagate those statistics into the function as well.
 *   * Output size inference happens for DML-bodied functions that are invoked exactly once
 *     and for external functions that are known in advance (see UDFs in packagesupport).
 *   * Size propagation across DAGs requires control flow awareness:
 *     - Generic statement blocks: updated variables -> old stats in; new stats out
 *     - While/for statement blocks: updated variables -> old stats in/out if loop insensitive; otherwise unknown
 *     - If statement blocks: updated variables -> old stats in; new stats out if branch-insensitive            
 *     
 *         
 */
public class InterProceduralAnalysis 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final boolean LDEBUG = false; //internal local debug level
	private static final Log LOG = LogFactory.getLog(InterProceduralAnalysis.class.getName());
    
	//internal configuration parameters
	private static final boolean INTRA_PROCEDURAL_ANALYSIS      = true; //propagate statistics across statement blocks (main/functions)	
	private static final boolean PROPAGATE_KNOWN_UDF_STATISTICS = true; //propagate statistics for known external functions 
	private static final boolean ALLOW_MULTIPLE_FUNCTION_CALLS  = true; //propagate consistent statistics from multiple calls 
	private static final boolean REMOVE_UNUSED_FUNCTIONS        = true; //removed unused functions (inlined or never called)
	
	static{
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("com.ibm.bi.dml.parser.InterProceduralAnalysis")
				  .setLevel((Level) Level.DEBUG);
		}
	}
	
	public InterProceduralAnalysis()
	{
		//do nothing
	}
	
	/**
	 * Public interface of IPA - everything else is meant for internal use only.
	 * 
	 * @param dmlt
	 * @param dmlp
	 * @throws HopsException
	 * @throws ParseException
	 * @throws LanguageException
	 */
	public void analyzeProgram( DMLTranslator dmlt, DMLProgram dmlp ) 
		throws HopsException, ParseException, LanguageException
	{
		//step 1: get candidates for statistics propagation into functions (if required)
		HashMap<String, Integer> fcandCounts = new HashMap<String, Integer>();
		HashMap<String, FunctionOp> fcandHops = new HashMap<String, FunctionOp>();
		HashMap<String, HashSet<Long>> fcandSafeNNZ = new HashMap<String, HashSet<Long>>(); 
		HashSet<String> allFCandKeys = new HashSet<String>();
		if( dmlp.getFunctionStatementBlocks().size() > 0 )
		{
			for ( StatementBlock sb : dmlp.getStatementBlocks() ) //get candidates (over entire program)
				getFunctionCandidatesForStatisticPropagation( sb, fcandCounts, fcandHops );
			allFCandKeys.addAll(fcandCounts.keySet()); //cp before pruning
			pruneFunctionCandidatesForStatisticPropagation( fcandCounts, fcandHops );	
			determineFunctionCandidatesNNZPropagation( fcandHops, fcandSafeNNZ );
			dmlt.resetHopsDAGVisitStatus( dmlp );
		}
		
		if( fcandCounts.size()>0 || INTRA_PROCEDURAL_ANALYSIS ) {
			//step 2: propagate statistics into functions and across DAGs
			//(callVars used to chain outputs/inputs of multiple functions calls) 
			LocalVariableMap callVars = new LocalVariableMap();
			for ( StatementBlock sb : dmlp.getStatementBlocks() ) //propagate stats into candidates
				propagateStatisticsAcrossBlock( sb, fcandCounts.keySet(), callVars, fcandSafeNNZ, new HashSet<String>() );
		}
		
		//step 3: remove unused functions (e.g., inlined or never called)
		if( REMOVE_UNUSED_FUNCTIONS ) {
			removeUnusedFunctions( dmlp, allFCandKeys );
		}
	}
	
	
	/////////////////////////////
	// GET FUNCTION CANDIDATES
	//////
	
	/**
	 * 
	 * @param sb
	 * @param fcand
	 * @throws HopsException
	 * @throws ParseException
	 */
	private void getFunctionCandidatesForStatisticPropagation( StatementBlock sb, HashMap<String, Integer> fcandCounts, HashMap<String, FunctionOp> fcandHops ) 
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
	
	/**
	 * 
	 * @param prog
	 * @param hop
	 * @param fcand
	 * @throws HopsException
	 * @throws ParseException
	 */
	private void getFunctionCandidatesForStatisticPropagation(DMLProgram prog, Hop hop, HashMap<String, Integer> fcandCounts, HashMap<String, FunctionOp> fcandHops ) 
		throws HopsException, ParseException
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
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
								   &&  h1.get_dim1()==h2.get_dim1() 
								   &&  h1.get_dim2()==h2.get_dim2()
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
		
		hop.set_visited(VISIT_STATUS.DONE);
	}
	
	/**
	 * 
	 * @param fcand
	 */
	private void pruneFunctionCandidatesForStatisticPropagation(HashMap<String, Integer> fcandCounts, HashMap<String, FunctionOp> fcandHops)
	{
		//debug input
		if( LOG.isDebugEnabled() )
			for( String key : fcandCounts.keySet() )
			{
				LOG.debug("IPA: FUNC statistic propagation candidate: "+key+", callCount="+fcandCounts.get(key));
			}
		
		//materialize key set
		HashSet<String> tmp = new HashSet<String>();
		for( String key : fcandCounts.keySet() )
			tmp.add(key);
		
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
				//System.out.println("IPA: FUNC statistic propagation candidate (after pruning): "+key);
			}
	}

	/////////////////////////////
	// DETERMINE NNZ PROPAGATE SAFENESS
	//////

	/**
	 * Populates fcandSafeNNZ with all <functionKey,hopID> pairs where it is safe to
	 * propagate nnz into the function.
	 *  
	 * @param fcandHops
	 * @param fcandSafeNNZ
	 */
	private void determineFunctionCandidatesNNZPropagation(HashMap<String, FunctionOp> fcandHops, HashMap<String, HashSet<Long>> fcandSafeNNZ)
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
	 * 
	 * @param sb
	 * @param fcand
	 * @throws HopsException
	 * @throws ParseException
	 * @throws CloneNotSupportedException 
	 */
	private void propagateStatisticsAcrossBlock( StatementBlock sb, Set<String> fcand, LocalVariableMap callVars, HashMap<String, HashSet<Long>> fcandSafeNNZ, HashSet<String> fnStack ) 
		throws HopsException, ParseException
	{
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sbi : fstmt.getBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, fnStack);
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
				propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, fnStack);
			if( Recompiler.reconcileUpdatedCallVarsLoops(oldCallVars, callVars, wsb) ){ //second pass if required
				propagateStatisticsAcrossPredicateDAG(wsb.getPredicateHops(), callVars);
				for (StatementBlock sbi : wstmt.getBody())
					propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, fnStack);
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
				propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, fnStack);
			for (StatementBlock sbi : istmt.getElseBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVarsElse, fcandSafeNNZ, fnStack);
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
				propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, fnStack);
			if( Recompiler.reconcileUpdatedCallVarsLoops(oldCallVars, callVars, fsb) )
				for (StatementBlock sbi : fstmt.getBody())
					propagateStatisticsAcrossBlock(sbi, fcand, callVars, fcandSafeNNZ, fnStack);
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
			//refresh stats across dag
			Hop.resetVisitStatus(roots);
			propagateStatisticsAcrossDAG(roots, callVars);
			//propagate stats into function calls
			Hop.resetVisitStatus(roots);
			propagateStatisticsIntoFunctions(prog, roots, fcand, callVars, fcandSafeNNZ, fnStack);
		}
	}
	

	/**
	 * 
	 * @param root
	 * @param vars
	 * @throws HopsException
	 */
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
	 * 
	 * @param roots
	 * @param vars
	 * @throws HopsException
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
	 * 
	 * @param prog
	 * @param hop
	 * @param fcand
	 * @param callVars
	 * @throws HopsException
	 * @throws ParseException
	 */
	private void propagateStatisticsIntoFunctions(DMLProgram prog, ArrayList<Hop> roots, Set<String> fcand, LocalVariableMap callVars, HashMap<String, HashSet<Long>> fcandSafeNNZ, HashSet<String> fnStack ) 
			throws HopsException, ParseException
	{
		for( Hop root : roots )
			propagateStatisticsIntoFunctions(prog, root, fcand, callVars, fcandSafeNNZ, fnStack);
	}
	
	
	/**
	 * 
	 * @param prog
	 * @param hop
	 * @param fcand
	 * @throws HopsException
	 * @throws ParseException
	 */
	private void propagateStatisticsIntoFunctions(DMLProgram prog, Hop hop, Set<String> fcand, LocalVariableMap callVars, HashMap<String, HashSet<Long>> fcandSafeNNZ, HashSet<String> fnStack ) 
		throws HopsException, ParseException
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		for( Hop c : hop.getInput() )
			propagateStatisticsIntoFunctions(prog, c, fcand, callVars, fcandSafeNNZ, fnStack);
		
		if( hop instanceof FunctionOp )
		{
			//maintain counters and investigate functions if not seen so far
			FunctionOp fop = (FunctionOp) hop;
			String fkey = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(), fop.getFunctionName());
			if( fcand.contains(fkey) && 
				!fnStack.contains(fkey) &&  //prevent recursion	
			    fop.getFunctionType() == FunctionType.DML )
			{
				//maintain function call stack
				fnStack.add(fkey);
				
				//propagate statistics
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				
				//create mapping and populate symbol table for refresh
				LocalVariableMap tmpVars = new LocalVariableMap();
				populateLocalVariableMapForFunctionCall( fstmt, fop, tmpVars, fcandSafeNNZ.get(fkey) );

				//recursively propagate statistics
				propagateStatisticsAcrossBlock(fsb, fcand, tmpVars, fcandSafeNNZ, fnStack);
				
				//extract vars from symbol table, re-map and refresh main program
				extractFunctionCallReturnStatistics(fstmt, fop, tmpVars, callVars, true);		
				
				//maintain function call stack
				fnStack.remove(fkey);
			}
			else if (   fop.getFunctionType() == FunctionType.EXTERNAL_FILE
				     || fop.getFunctionType() == FunctionType.EXTERNAL_MEM  )
			{
				//infer output size for known external functions
				if( PROPAGATE_KNOWN_UDF_STATISTICS ) {
					FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
					ExternalFunctionStatement fstmt = (ExternalFunctionStatement) fsb.getStatement(0);
					extractExternalFunctionCallReturnStatistics(fstmt, fop, callVars);
				}
			}
		}
		
		hop.set_visited(VISIT_STATUS.DONE);
	}
	
	
	/**
	 * 
	 * @param fstmt
	 * @param fop
	 * @param vars
	 * @throws HopsException 
	 */
	private void populateLocalVariableMapForFunctionCall( FunctionStatement fstmt, FunctionOp fop, LocalVariableMap vars, HashSet<Long> inputSafeNNZ ) 
		throws HopsException
	{
		Vector<DataIdentifier> inputVars = fstmt.getInputParams();
		ArrayList<Hop> inputOps = fop.getInput();
		
		for( int i=0; i<inputVars.size(); i++ )
		{
			//create mapping between input hops and vars
			DataIdentifier dat = inputVars.get(i);
			Hop input = inputOps.get(i);
			
			if( input.get_dataType()==DataType.MATRIX )
			{
				//propagate matrix characteristics
				MatrixObject mo = new MatrixObject(ValueType.DOUBLE, null);
				MatrixCharacteristics mc = new MatrixCharacteristics( 
											input.get_dim1(), input.get_dim2(), 
											DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize,
											inputSafeNNZ.contains(input.getHopID())?input.getNnz():-1 );
				MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,null,null);
				mo.setMetaData(meta);	
				vars.put(dat.getName(), mo);	
			}
			else if( input.get_dataType()==DataType.SCALAR 
					&& input instanceof LiteralOp           )
			{
				//propagate literal scalars into functions
				LiteralOp lit = (LiteralOp)input;
				ScalarObject scalar = null;
				switch(input.get_valueType())
				{
					case DOUBLE:	scalar = new DoubleObject(lit.getDoubleValue()); break;
					case INT:		scalar = new IntObject((int) lit.getLongValue()); break;
					case BOOLEAN: 	scalar = new BooleanObject(lit.getBooleanValue()); break;
					case STRING:	scalar = new StringObject(lit.getStringValue()); break;
				}
				vars.put(dat.getName(), scalar);	
			}
			
		}
	}
	
	/**
	 * 
	 * @param fstmt
	 * @param fop
	 * @param tmpVars
	 * @param callVars
	 * @param overwrite
	 * @throws HopsException
	 */
	private void extractFunctionCallReturnStatistics( FunctionStatement fstmt, FunctionOp fop, LocalVariableMap tmpVars, LocalVariableMap callVars, boolean overwrite ) 
		throws HopsException
	{
		Vector<DataIdentifier> foutputOps = fstmt.getOutputParams();
		String[] outputVars = fop.getOutputVariableNames();
		String fkey = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(), fop.getFunctionName());
		
		try
		{
			for( int i=0; i<foutputOps.size(); i++ )
			{
				DataIdentifier di = foutputOps.get(i);
				String fvarname = di.getName(); //name in function signature
				String pvarname = outputVars[i]; //name in calling program
				
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
							MatrixCharacteristics mc = ((MatrixFormatMetaData)moOut.getMetaData()).getMatrixCharacteristics();
							if( OptimizerUtils.estimateSizeExactSparsity(mc.get_rows(), mc.get_cols(), (mc.getNonZeros()>0)?((double)mc.getNonZeros())/mc.get_rows()/mc.get_cols():1.0)	
							    < OptimizerUtils.estimateSize(moIn.getNumRows(), moIn.getNumColumns(), 1.0d) )
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
	
	/**
	 * 
	 * @param fstmt
	 * @param fop
	 * @param callVars
	 * @throws HopsException
	 */
	private void extractExternalFunctionCallReturnStatistics( ExternalFunctionStatement fstmt, FunctionOp fop, LocalVariableMap callVars ) 
		throws HopsException
	{
		String className = fstmt.getOtherParams().get(ExternalFunctionStatement.CLASS_NAME);

		if(    className.equals(OrderWrapper.class.getName()) 
			|| className.equals(DeNaNWrapper.class.getCanonicalName())
			|| className.equals(DeNegInfinityWrapper.class.getCanonicalName()) )
		{
			Hop input = fop.getInput().get(0);
			long lnnz = className.equals(OrderWrapper.class.getName()) ? input.getNnz() : -1;
			MatrixObject moOut = createOutputMatrix(input.get_dim1(), input.get_dim2(),lnnz);
			callVars.put(fop.getOutputVariableNames()[0], moOut);
		}
		else if( className.equals("com.ibm.bi.dml.packagesupport.EigenWrapper") ) 
		//else if( className.equals(EigenWrapper.class.getName()) ) //string ref for build flexibility
		{
			Hop input = fop.getInput().get(0);
			callVars.put(fop.getOutputVariableNames()[0], createOutputMatrix(input.get_dim1(), 1, -1));
			callVars.put(fop.getOutputVariableNames()[1], createOutputMatrix(input.get_dim1(), input.get_dim1(),-1));			
		}
		else if( className.equals("com.ibm.bi.dml.packagesupport.LinearSolverWrapperCP") ) 
		//else if( className.equals(LinearSolverWrapperCP.class.getName()) ) //string ref for build flexibility
		{
			Hop input = fop.getInput().get(1);
			callVars.put(fop.getOutputVariableNames()[0], createOutputMatrix(input.get_dim1(), 1, -1));
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
	}
	
	/**
	 * 
	 * @param dim1
	 * @param dim2
	 * @param nnz
	 * @return
	 */
	private MatrixObject createOutputMatrix( long dim1, long dim2, long nnz )
	{
		MatrixObject moOut = new MatrixObject(ValueType.DOUBLE, null);
		MatrixCharacteristics mc = new MatrixCharacteristics( 
									dim1, dim2,
									DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize,
									nnz);
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,null,null);
		moOut.setMetaData(meta);
		
		return moOut;
	}
	
	/////////////////////////////
	// REMOVE UNUSED FUNCTIONS
	//////

	/**
	 * 
	 * @param dmlp
	 * @param fcandKeys
	 * @throws LanguageException 
	 */
	public void removeUnusedFunctions( DMLProgram dmlp, Set<String> fcandKeys )
		throws LanguageException
	{
		Set<String> fnamespaces = dmlp.getNamespaces().keySet();
		for( String fnspace : fnamespaces  )
		{
			HashMap<String, FunctionStatementBlock> fsbs = dmlp.getFunctionStatementBlocks(fnspace);
			Iterator<Entry<String, FunctionStatementBlock>> iter = fsbs.entrySet().iterator();
			while( iter.hasNext() )
			{
				Entry<String, FunctionStatementBlock> e = iter.next();
				String fname = e.getKey();
				String fKey = DMLProgram.constructFunctionKey(fnspace, fname);
				//probe function candidates, remove if no candidate
				if( !fcandKeys.contains(fKey) )
					iter.remove();
			}
		}
	}
}
