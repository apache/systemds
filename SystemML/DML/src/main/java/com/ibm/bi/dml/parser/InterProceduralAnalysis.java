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
import com.ibm.bi.dml.runtime.controlprogram.Program;
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
 *      also includes control-flow aware propagation of size and sparsity.
 * 
 * In general, the basic concepts of IPA are as follows and all places that deal with
 * statistic propagation should adhere to that:
 *   * Rule 1: Exact size propagation: Since the dimension information are sometimes used
 *     for specific lops construction (e.g., in append), we cannot propagate worst-case 
 *     estimates but only exact information; otherwise size must be unknown.
 *   * Rule 2: Dimension information and sparsity are handled separately, i.e., if an updated 
 *     variable has changing sparsity but constant dimensions, its dimensions are known but
 *     sparsity unknown.
 * 
 * More specifically, those two rules are currently realized as follows:
 *   * Statistics propagation is applied for DML-bodied functions that are invoked exactly once.
 *     This ensures that we can savely propagate exact information into this function.
 *   * Output size inference happens for DML-bodied functions that are invoked exactly once
 *     and for external functions that are known in advance (see UDFs in packagesupport).
 *   * Size propagation across DAGs requires control flow awareness:
 *     - Generic statement blocks: updated variables -> old stats in; new stats out
 *     - While/for statement blocks: updated variables -> old stats in/out if loop insensitive; otherwise unknown
 *     - If statement blocks: updated variables -> old stats in; new stats out if branch-insensitive            
 * 
 * TODO Additional potential for improvement
 *   * Multiple function calls with equivalent input characteristics: Propagate statistics
 *     into functions even if called multiple times as long as all input characteristics are
 *     equivalent. However, propagating worst-case statistics is currently not applicable 
 *     because we distinguish between exact and worst-case size information. 
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
    
	//internal parameters
	private static final boolean INTRA_PROCEDURAL_ANALYSIS = true;
	private static final boolean PROPAGATE_KNOWN_UDF_STATISTICS = true;
	
	
	static{
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("com.ibm.bi.dml.parser.InterProceduralAnalysis")
				  .setLevel((Level) Level.TRACE);
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
		HashMap<String, Integer> fcand = new HashMap<String, Integer>();
		if( dmlp.getFunctionStatementBlocks().size() > 0 )
		{
			for ( StatementBlock sb : dmlp.getStatementBlocks() ) //get candidates (over entire program)
				getFunctionCandidatesForStatisticPropagation( sb, fcand );
			pruneFunctionCandidatesForStatisticPropagation( fcand );	
			dmlt.resetHopsDAGVisitStatus( dmlp );
		}
		
		if( fcand.size()>0 || INTRA_PROCEDURAL_ANALYSIS ) {
			//step 2: propagate statistics into functions and across DAGs
			//(callVars used to chain outputs/inputs of multiple functions calls) 
			LocalVariableMap callVars = new LocalVariableMap();
			for ( StatementBlock sb : dmlp.getStatementBlocks() ) //propagate stats into candidates
				propagateStatisticsAcrossBlock( sb, fcand.keySet(), callVars );
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
	private void getFunctionCandidatesForStatisticPropagation( StatementBlock sb, HashMap<String, Integer> fcand ) 
		throws HopsException, ParseException
	{
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sbi : fstmt.getBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcand);
		}
		else if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			for (StatementBlock sbi : wstmt.getBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcand);
		}	
		else if (sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			for (StatementBlock sbi : istmt.getIfBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcand);
			for (StatementBlock sbi : istmt.getElseBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcand);
		}
		else if (sb instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			for (StatementBlock sbi : fstmt.getBody())
				getFunctionCandidatesForStatisticPropagation(sbi, fcand);
		}
		else //generic (last-level)
		{
			ArrayList<Hop> roots = sb.get_hops();
			for( Hop root : roots )
				getFunctionCandidatesForStatisticPropagation(sb.getDMLProg(), root, fcand);
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
	private void getFunctionCandidatesForStatisticPropagation(DMLProgram prog, Hop hop, HashMap<String, Integer> fcand ) 
		throws HopsException, ParseException
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		if( hop instanceof FunctionOp )
		{
			//maintain counters and investigate functions if not seen so far
			FunctionOp fop = (FunctionOp) hop;
			String fkey = fop.getFunctionNamespace() + Program.KEY_DELIM + fop.getFunctionName();
			if( fcand.containsKey(fkey) )
			{
				//maintain counter (this function is no candidate)
				fcand.put(fkey, fcand.get(fkey)+1);
			}
			else
			{
				fcand.put(fkey, 1); //create a new entry
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
				getFunctionCandidatesForStatisticPropagation(fsb, fcand);
			}
		}
			
		for( Hop c : hop.getInput() )
			getFunctionCandidatesForStatisticPropagation(prog, c, fcand);
		
		hop.set_visited(VISIT_STATUS.DONE);
	}
	
	/**
	 * 
	 * @param fcand
	 */
	private void pruneFunctionCandidatesForStatisticPropagation(HashMap<String, Integer> fcand)
	{
		//debug input
		if( LOG.isDebugEnabled() )
			for( String key : fcand.keySet() )
			{
				LOG.debug("IPA: FUNC statistic propagation candidate: "+key+", callCount="+fcand.get(key));
			}
		
		//materialize key set
		HashSet<String> tmp = new HashSet<String>();
		for( String key : fcand.keySet() )
			tmp.add(key);
		
		//check and prune candidate list
		for( String key : tmp )
		{
			Integer cnt = fcand.get(key);
			if( cnt != null && cnt > 1 ) //if multiple refs
				fcand.remove(key);
		}
		
		//debug output
		if( LOG.isDebugEnabled() )
			for( String key : fcand.keySet() )
			{
				LOG.debug("IPA: FUNC statistic propagation candidate (after pruning): "+key);
				//System.out.println("IPA: FUNC statistic propagation candidate (after pruning): "+key);
			}
	}

	
	/////////////////////////////
	// INTRA-PROCEDURE ANALYIS
	//////	
	
	/**
	 * 
	 * @param sb
	 * @param fcand
	 * @throws HopsException
	 * @throws ParseException
	 * @throws CloneNotSupportedException 
	 */
	private void propagateStatisticsAcrossBlock( StatementBlock sb, Set<String> fcand, LocalVariableMap callVars ) 
		throws HopsException, ParseException
	{
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock sbi : fstmt.getBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVars);
		}
		else if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			//old stats into predicate
			propagateStatisticsAcrossPredicateDAG(wsb.getPredicateHops(), callVars);
			//check and propagate stats into body
			LocalVariableMap oldCallVars = (LocalVariableMap) callVars.clone();
			for (StatementBlock sbi : wstmt.getBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVars);
			if( reconcileUpdatedCallVarsLoops(oldCallVars, callVars, wsb) ) //second pass if required
				for (StatementBlock sbi : wstmt.getBody())
					propagateStatisticsAcrossBlock(sbi, fcand, callVars);
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
				propagateStatisticsAcrossBlock(sbi, fcand, callVars);
			for (StatementBlock sbi : istmt.getElseBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVarsElse);
			callVars = reconcileUpdatedCallVarsIf(oldCallVars, callVars, callVarsElse, isb);
		}
		else if (sb instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			//old stats into predicate
			propagateStatisticsAcrossPredicateDAG(fsb.getFromHops(), callVars);
			propagateStatisticsAcrossPredicateDAG(fsb.getToHops(), callVars);
			propagateStatisticsAcrossPredicateDAG(fsb.getIncrementHops(), callVars);
			//check and propagate stats into body
			LocalVariableMap oldCallVars = (LocalVariableMap) callVars.clone();
			for (StatementBlock sbi : fstmt.getBody())
				propagateStatisticsAcrossBlock(sbi, fcand, callVars);
			if( reconcileUpdatedCallVarsLoops(oldCallVars, callVars, fsb) )
				for (StatementBlock sbi : fstmt.getBody())
					propagateStatisticsAcrossBlock(sbi, fcand, callVars);	
		}
		else //generic (last-level)
		{
			//old stats in, new stats out if updated
			ArrayList<Hop> roots = sb.get_hops();
			DMLProgram prog = sb.getDMLProg();
			//refresh stats across dag
			Hop.resetVisitStatus(roots);
			propagateStatisticsAcrossDAG(roots, callVars);
			//propagate stats into function calls
			Hop.resetVisitStatus(roots);
			propagateStatisticsIntoFunctions(prog, roots, fcand, callVars);
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
	
	/**
	 * 
	 * @param oldCallVars
	 * @param callVars
	 * @param sb
	 * @return
	 */
	private boolean reconcileUpdatedCallVarsLoops( LocalVariableMap oldCallVars, LocalVariableMap callVars, StatementBlock sb )
	{
		boolean requiresRecompile = false;
		for( String varname : sb._updated.getVariableNames() )
		{
			Data dat1 = oldCallVars.get(varname);
			Data dat2 = callVars.get(varname);
			if( dat1!=null && dat1 instanceof MatrixObject && dat2!=null && dat2 instanceof MatrixObject )
			{
				MatrixObject moOld = (MatrixObject) dat1;
				MatrixObject mo = (MatrixObject) dat2;
				MatrixCharacteristics mcOld = ((MatrixFormatMetaData)moOld.getMetaData()).getMatrixCharacteristics();
				MatrixCharacteristics mc = ((MatrixFormatMetaData)mo.getMetaData()).getMatrixCharacteristics();
				
				if( mcOld.get_rows() != mc.get_rows() 
					|| mcOld.get_cols() != mc.get_cols()
					|| mcOld.getNonZeros() != mc.getNonZeros() )
				{
					long ldim1 =mc.get_rows(), ldim2=mc.get_cols(), lnnz=mc.getNonZeros();
					//handle dimension change in body
					if(    mcOld.get_rows() != mc.get_rows() 
						|| mcOld.get_cols() != mc.get_cols() )
					{
						ldim1=-1;
						ldim2=-1; //unknown
						requiresRecompile = true;
					}
					//handle sparsity change
					if( mcOld.getNonZeros() != mc.getNonZeros() )
					{
						lnnz=-1; //unknown		
						requiresRecompile = true;
					}
					
					
					MatrixObject moNew = createOutputMatrix(ldim1, ldim2, lnnz);
					callVars.put(varname, moNew);
				}
			}
		}
		
		return requiresRecompile;
	}

	/**
	 * 
	 * @param oldCallVars
	 * @param callVarsIf
	 * @param callVarsElse
	 * @param sb
	 * @return
	 */
	private LocalVariableMap reconcileUpdatedCallVarsIf( LocalVariableMap oldCallVars, LocalVariableMap callVarsIf, LocalVariableMap callVarsElse, StatementBlock sb )
	{
		for( String varname : sb._updated.getVariableNames() )
		{	
			Data origVar = oldCallVars.get(varname);
			Data ifVar = callVarsIf.get(varname);
			Data elseVar = callVarsElse.get(varname);
			Data dat1 = null, dat2 = null;
			
			if( ifVar!=null && elseVar!=null ){ // both branches exists
				dat1 = ifVar;
				dat2 = elseVar;
			}
			else if( ifVar!=null && elseVar==null ){ //only if
				dat1 = origVar;
				dat2 = ifVar;
			}
			else { //only else
				dat1 = origVar;
				dat2 = elseVar;
			}
			
			if( dat1 != null && dat1 instanceof MatrixObject && dat2!=null && dat2 instanceof MatrixObject )
			{
				MatrixObject moOld = (MatrixObject) dat1;
				MatrixObject mo = (MatrixObject) dat2;
				MatrixCharacteristics mcOld = ((MatrixFormatMetaData)moOld.getMetaData()).getMatrixCharacteristics();
				MatrixCharacteristics mc = ((MatrixFormatMetaData)mo.getMetaData()).getMatrixCharacteristics();
				
				if( mcOld.get_rows() != mc.get_rows() 
						|| mcOld.get_cols() != mc.get_cols()
						|| mcOld.getNonZeros() != mc.getNonZeros() )
				{
					long ldim1 =mc.get_rows(), ldim2=mc.get_cols(), lnnz=mc.getNonZeros();
					
					//handle dimension change
					if(    mcOld.get_rows() != mc.get_rows() 
						|| mcOld.get_cols() != mc.get_cols() )
					{
						ldim1=-1; ldim2=-1; //unknown
					}
					//handle sparsity change
					if( mcOld.getNonZeros() != mc.getNonZeros() )
					{
						lnnz=-1; //unknown		
					}
					
					MatrixObject moNew = createOutputMatrix(ldim1, ldim2, lnnz);
					callVarsIf.put(varname, moNew);
				}
			}
		}
		
		return callVarsIf;
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
	private void propagateStatisticsIntoFunctions(DMLProgram prog, ArrayList<Hop> roots, Set<String> fcand, LocalVariableMap callVars ) 
			throws HopsException, ParseException
	{
		for( Hop root : roots )
			propagateStatisticsIntoFunctions(prog, root, fcand, callVars);
	}
	
	
	/**
	 * 
	 * @param prog
	 * @param hop
	 * @param fcand
	 * @throws HopsException
	 * @throws ParseException
	 */
	private void propagateStatisticsIntoFunctions(DMLProgram prog, Hop hop, Set<String> fcand, LocalVariableMap callVars ) 
		throws HopsException, ParseException
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		for( Hop c : hop.getInput() )
			propagateStatisticsIntoFunctions(prog, c, fcand, callVars);
		
		if( hop instanceof FunctionOp )
		{
			//maintain counters and investigate functions if not seen so far
			FunctionOp fop = (FunctionOp) hop;
			String fkey = fop.getFunctionNamespace() + Program.KEY_DELIM + fop.getFunctionName();
			if( fcand.contains(fkey) &&
			    fop.getFunctionType() == FunctionType.DML )
			{
				//propagate statistics
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				
				//create mapping and populate symbol table for refresh
				LocalVariableMap tmpVars = new LocalVariableMap();
				populateLocalVariableMapForFunctionCall( fstmt, fop, tmpVars );

				//recursively propagate statistics
				propagateStatisticsAcrossBlock(fsb, fcand, tmpVars);
				
				//extract vars from symbol table, re-map and refresh main program
				extractFunctionCallReturnStatistics(fstmt, fop, tmpVars, callVars, true);				
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
	private void populateLocalVariableMapForFunctionCall( FunctionStatement fstmt, FunctionOp fop, LocalVariableMap vars ) 
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
											input.getNnz() );
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
		String fkey = fop.getFunctionNamespace() + Program.KEY_DELIM + fop.getFunctionName();
		
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
		/*else if( className.equals(EigenWrapper.class.getName()) ) 
		{
			Hop input = fop.getInput().get(0);
			callVars.put(fop.getOutputVariableNames()[0], createOutputMatrix(input.get_dim1(), input.get_dim1(),-1));
			callVars.put(fop.getOutputVariableNames()[1], createOutputMatrix(input.get_dim1(), input.get_dim1(),-1));			
		}
		else if( className.equals(LinearSolverWrapperCP.class.getName()) ) 
		{
			Hop input = fop.getInput().get(1);
			callVars.put(fop.getOutputVariableNames()[0], createOutputMatrix(input.get_dim1(), 1, -1));
		}*/
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
}
