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

package org.apache.sysds.parser;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.parser.Expression.BinaryOp;
import org.apache.sysds.parser.PrintStatement.PRINTTYPE;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PResultMerge;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PartitionFormat;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * This ParForStatementBlock is essentially identical to a ForStatementBlock, except an extended validate
 * for checking/setting optional parfor parameters and running the loop dependency analysis.
 * 
 */
public class ParForStatementBlock extends ForStatementBlock 
{
	private static final boolean LDEBUG = false; //internal local debug level
	protected static final Log LOG = LogFactory.getLog(ParForStatementBlock.class.getName());
	
	//external parameter names 
	private static HashSet<String> _paramNames;
	public static final String CHECK            = "check";       //run loop dependency analysis
	public static final String PAR              = "par";         //number of parallel workers
	public static final String TASK_PARTITIONER = "taskpartitioner"; //task partitioner
	public static final String TASK_SIZE        = "tasksize";    //number of tasks 
	public static final String DATA_PARTITIONER = "datapartitioner"; //task partitioner 
	public static final String RESULT_MERGE     = "resultmerge"; //task partitioner 
	public static final String EXEC_MODE        = "mode";        //runtime execution mode
	public static final String OPT_MODE         = "opt";        //runtime execution mode
	public static final String OPT_LOG          = "log";        //parfor logging mode
	public static final String PROFILE          = "profile";    //monitor and report parfor performance profile
	
	//default external parameter values
	private static HashMap<String, String> _paramDefaults;
	private static HashMap<String, String> _paramDefaults2; //for constrained opt
	
	//internal parameter values
	private static final boolean NORMALIZE                 = false; //normalize FOR from to incr
	private static final boolean USE_FN_CACHE              = false; //useful for larger scripts (due to O(n^2))
	private static final boolean ABORT_ON_FIRST_DEPENDENCY = true;
	private static final boolean CONSERVATIVE_CHECK        = false; //include FOR into dep analysis, reject unknown vars (otherwise use internal vars for whole row or column)
	
	public static final String INTERAL_FN_INDEX_ROW       = "__ixr"; //pseudo index for range indexing row
	public static final String INTERAL_FN_INDEX_COL       = "__ixc"; //pseudo index for range indexing col 
	
	private static final IDSequence _idSeq = new IDSequence();
	private static final IDSequence _idSeqfn = new IDSequence();
	
	private static HashMap<String, LinearFunction> _fncache; //slower for most (small cases) cases
	
	//instance members
	private final long _PID;
	private VariableSet          _vsParent   = null;
	private ArrayList<ResultVar> _resultVars = null;
	private Bounds               _bounds     = null;
	
	static
	{
		// populate parameter name lookup-table
		_paramNames = new HashSet<>();
		_paramNames.add( CHECK );
		_paramNames.add( PAR );
		_paramNames.add( TASK_PARTITIONER );
		_paramNames.add( TASK_SIZE );
		_paramNames.add( DATA_PARTITIONER );
		_paramNames.add( RESULT_MERGE );
		_paramNames.add( EXEC_MODE );
		_paramNames.add( OPT_MODE );
		_paramNames.add( PROFILE );
		_paramNames.add( OPT_LOG );
		
		// populate defaults lookup-table
		_paramDefaults = new HashMap<>();
		_paramDefaults.put( CHECK,             "1" );
		_paramDefaults.put( PAR,               String.valueOf(InfrastructureAnalyzer.getLocalParallelism()) );
		_paramDefaults.put( TASK_PARTITIONER,  String.valueOf(PTaskPartitioner.FIXED) );
		_paramDefaults.put( TASK_SIZE,         "1" );
		_paramDefaults.put( DATA_PARTITIONER,  String.valueOf(PDataPartitioner.NONE) );
		_paramDefaults.put( RESULT_MERGE,      String.valueOf(PResultMerge.LOCAL_AUTOMATIC) );
		_paramDefaults.put( EXEC_MODE,         String.valueOf(PExecMode.LOCAL) );
		_paramDefaults.put( OPT_MODE,          String.valueOf(POptMode.RULEBASED) );
		_paramDefaults.put( PROFILE,           "0" );
		_paramDefaults.put( OPT_LOG,           OptimizerUtils.getDefaultLogLevel().toString() );
		
		_paramDefaults2 = new HashMap<>(); //OPT_MODE always specified
		_paramDefaults2.put( CHECK,            "1" );
		_paramDefaults2.put( PAR,              "-1" );
		_paramDefaults2.put( TASK_PARTITIONER, String.valueOf(PTaskPartitioner.UNSPECIFIED) );
		_paramDefaults2.put( TASK_SIZE,        "-1" );
		_paramDefaults2.put( DATA_PARTITIONER, String.valueOf(PDataPartitioner.UNSPECIFIED) );
		_paramDefaults2.put( RESULT_MERGE,     String.valueOf(PResultMerge.UNSPECIFIED) );
		_paramDefaults2.put( EXEC_MODE,        String.valueOf(PExecMode.UNSPECIFIED) );
		_paramDefaults2.put( PROFILE,          "0" );
		_paramDefaults2.put( OPT_LOG,          OptimizerUtils.getDefaultLogLevel().toString() );
		
		//initialize function cache
		if( USE_FN_CACHE ) {
			_fncache = new HashMap<>();
		}
		
		// for internal debugging only
		if( LDEBUG ) {
			Logger.getLogger("org.apache.sysds.parser.ParForStatementBlock")
				.setLevel(Level.TRACE);
		}
	}
	
	public ParForStatementBlock() {
		_PID = _idSeq.getNextID();
		_resultVars = new ArrayList<>();
		
		LOG.trace("PARFOR("+_PID+"): ParForStatementBlock instance created");
	}
	
	public long getID() {
		return _PID;
	}

	public ArrayList<ResultVar> getResultVariables() {
		return _resultVars;
	}
	
	public void setResultVariables(ArrayList<ResultVar> rvars) {
		_resultVars.clear();
		_resultVars.addAll(rvars);
	}
	
	private void addToResultVariablesNoDup( String var, boolean accum ) {
		addToResultVariablesNoDup(new ResultVar(var, accum));
	}
	
	private void addToResultVariablesNoDup( ResultVar var ) {
		if( !_resultVars.contains( var ) )
			_resultVars.add( var );
	}
	
	@Override
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars, boolean conditional)
	{
		LOG.trace("PARFOR("+_PID+"): validating ParForStatementBlock.");
		
		//create parent variable set via cloning
		_vsParent = new VariableSet( ids );
		
		if(LOG.isTraceEnabled()) //note: A is matrix, and A[i,1] is scalar
			for( DataIdentifier di : _vsParent.getVariables().values() )
				LOG.trace("PARFOR: non-local "+di._name+": "+di.getDataType().toString()+" with rowDim = "+di.getDim1());
		
		//normal validate via ForStatement (sequential)
		//NOTES:
		// * validate/dependency checking of nested parfor-loops happens at this point
		// * validate includes also constant propagation for from, to, incr expressions
		// * this includes also function inlining
		VariableSet vs = super.validate(dmlProg, ids, constVars, conditional);
		
		//check of correctness of specified parfor parameter names and 
		//set default parameter values for all not specified parameters 
		ParForStatement pfs = (ParForStatement) _statements.get(0);
		IterablePredicate predicate = pfs.getIterablePredicate();
		HashMap<String, String> params = predicate.getParForParams();
		if( params != null ) //if parameter specified
		{
			//check for valid parameter types
			for( String key : params.keySet() )
				if( !_paramNames.contains(key) ) //always unconditional
					raiseValidateError("PARFOR: The specified parameter '"+key+"' is no valid parfor parameter.", false);
			
			//set defaults for all non-specified values
			//(except if CONSTRAINT optimizer, in order to distinguish specified parameters)
			boolean constrained = (params.containsKey( OPT_MODE ) 
				&& params.get( OPT_MODE ).equalsIgnoreCase(POptMode.CONSTRAINED.name()));
			for( String key : _paramNames )
				if( !params.containsKey(key) )
				{
					if( constrained ) {
						params.put(key, _paramDefaults2.get(key));
					}
					else //default case
						params.put(key, _paramDefaults.get(key));
				}
			
			//keep info if forced into remote exec
			if( constrained && params.containsKey(EXEC_MODE) )
				dmlProg.setContainsRemoteParfor(
					params.get(EXEC_MODE).equals(PExecMode.REMOTE_SPARK.name()) ||
					params.get(EXEC_MODE).equals(PExecMode.REMOTE_SPARK_DP.name()));
		}
		else {
			//set all defaults
			params = new HashMap<>();
			params.putAll( _paramDefaults );
			predicate.setParForParams(params);
		}
		
		//start time measurement for normalization and dependency analysis
		Timing time = new Timing(true);
		
		// LOOP DEPENDENCY ANALYSIS (test for dependency existence)
		// no false negative guaranteed, but possibly false positives
		
		/* Basic intuition: WRITES to NON-local variables are only permitted iff
		 *   - no data dep (no read other than own iteration w i < r j)
		 *   - no anti dep (no read other than own iteration w i > r j)
		 *   - no output dep (no write other than own iteration)
		 *   
		 * ALGORITHM:
		 * 1) Determine candidates C (writes to non-local variables)
		 * 2) Prune all c from C where no dependencies --> C'
		 * 3) Raise an exception/warning if C' not the empty set 
		 * 
		 * RESTRICTIONS:
		 * - array subscripts of non-local variables must be linear functions of the form 
		 *   a0+ a1*i + ... + a2*j, where i and j are for or parfor indexes.
		 * - for and parfor increments must be integer values 
		 * - only static (integer lower, upper bounds) range indexing
		 * - only input variables considered as potential candidates for checking 
		 * 
		 *   (TODO: in order to remove the last restriction, dependencies must be checked again after 
		 *   live variable analysis against LIVEOUT)
		 * 
		 * NOTE: validity is only checked during compilation, i.e., for dynamic from, to, incr MIN MAX values assumed.
		 */ 
		
		LOG.trace("PARFOR: running loop dependency analysis ...");

		//### Step 1 ###: determine candidate set C
		HashSet<Candidate> C = new HashSet<>(); 
		HashSet<Candidate> C2 = new HashSet<>(); 
		Integer sCount = 0; //object for call by ref 
		rDetermineCandidates(pfs.getBody(), C, sCount);
		if( LOG.isTraceEnabled() )
			for(Candidate c : C)
				LOG.trace("PARFOR: dependency candidate: var '"+c._var+"' (accum="+c._isAccum+")");
		
		boolean check = (Integer.parseInt(params.get(CHECK))==1);
		if( check ) 
		{
			//### Step 2 ###: prune c without dependencies
			_bounds = new Bounds();
			for( FunctionStatementBlock fsb : dmlProg.getFunctionStatementBlocks() )
				rDetermineBounds( fsb, false ); //writes to _bounds	
			rDetermineBounds( dmlProg.getStatementBlocks(), false ); //writes to _bounds
			
			for( Candidate c : C )
			{
				DataType cdt = _vsParent.getVariables().get(c._var).getDataType(); //might be different in DataIdentifier
				
				//assume no dependency
				sCount = 0;
				boolean[] dep = new boolean[]{false,false,false}; //output, data, anti
				rCheckCandidates(c, cdt, pfs.getBody(), sCount, dep);
				
				if (LOG.isTraceEnabled()) {
					if( dep[0] ) 
						LOG.trace("PARFOR: output dependency detected for var '"+c._var+"'.");
					if( dep[1] ) 
						LOG.trace("PARFOR: data dependency detected for var '"+c._var+"'.");
					if( dep[2] ) 
						LOG.trace("PARFOR: anti dependency detected for var '"+c._var+"'.");
				}
				
				if( dep[0] || dep[1] || dep[2] ) {
					C2.add(c);
					if( ABORT_ON_FIRST_DEPENDENCY )
						break;
				}
			}

			
			//### Step 3 ###: raise an exception / warning
			if( C2.size() > 0 )
			{
				LOG.trace("PARFOR: loop dependencies detected.");

				StringBuilder depVars = new StringBuilder();
				for( Candidate c : C2 ) {
					if( depVars.length()>0 )
						depVars.append(", ");
					depVars.append(c._var);
				}
				
				//always unconditional (to ensure we always raise dependency issues)
				raiseValidateError("PARFOR loop dependency analysis: " +
					"inter-iteration (loop-carried) dependencies detected for variable(s): " +
					depVars.toString() +". \n " +
					"Please, ensure independence of iterations.", false);
			}
			else {
				LOG.trace("PARFOR: no loop dependencies detected.");
			}
		}
		else {
			LOG.debug("INFO: PARFOR("+_PID+"): loop dependency analysis skipped.");
		}
		
		//if successful, prepare result variables (all distinct vars in all candidates)
		//a) add own candidates
		for( Candidate var : C )
			if( check || var._dat.getDataType()!=DataType.SCALAR )
				addToResultVariablesNoDup( var._var, var._isAccum );
		//b) get and add child result vars (if required)
		ArrayList<ResultVar> tmp = new ArrayList<>();
		rConsolidateResultVars(pfs.getBody(), tmp);
		for( ResultVar var : tmp )
			if(_vsParent.containsVariable(var._name))
				addToResultVariablesNoDup(var);
		if( LOG.isDebugEnabled() )
			for( ResultVar rvar : _resultVars )
				LOG.debug("INFO: PARFOR final result variable: "+rvar._name);
		
		//cleanup function cache in order to prevent side effects between parfor statements
		if( USE_FN_CACHE )
			_fncache.clear();
		
		LOG.debug("INFO: PARFOR("+_PID+"): validate successful (no dependencies) in "+time.stop()+"ms.");

		//disable UMM if in effect and fallback to lazy write buffer
		if (OptimizerUtils.isUMMEnabled())
			OptimizerUtils.disableUMM();

		return vs;
	}
	
	public List<String> getReadOnlyParentMatrixVars() {
		VariableSet read = variablesRead();
		VariableSet updated = variablesUpdated();
		return liveIn().getVariableNames().stream() //read-only vars
			.filter(var -> read.containsVariable(var) && !updated.containsVariable(var))
			.filter(var -> read.isMatrix(var)).collect(Collectors.toList());
	}

	/**
	 * Determines the PDataPartitioningFormat for read-only parent variables according
	 * to the access pattern of that variable within the parfor statement block.
	 * Row-wise or column wise partitioning is only suggested if we see pure row-wise or
	 * column-wise access patterns.
	 * 
	 * @param var variables
	 * @return partition format
	 */
	public PartitionFormat determineDataPartitionFormat(String var) 
	{
		PartitionFormat dpf = null;
		List<PartitionFormat> dpfc = new LinkedList<>();
		
		try 
		{
			//determine partitioning candidates
			ParForStatement dpfs = (ParForStatement) _statements.get(0);
			rDeterminePartitioningCandidates(var, dpfs.getBody(), dpfc);
			
			//determine final solution		
			for( PartitionFormat tmp : dpfc )
				dpf = ( dpf!=null && !dpf.equals(tmp) ) ? //if no consensus
					PartitionFormat.NONE : tmp;
			if( dpf == null )
				dpf = PartitionFormat.NONE;
		}
		catch (LanguageException e) {
			LOG.trace( "Unable to determine partitioning candidates.", e );
			dpf = PartitionFormat.NONE;
		}
		
		return dpf;
	}
	
	/**
	 * This method recursively determines candidates for output,data,anti dependencies. 
	 * Candidates are defined as writes to non-local variables.
	 * 
	 * @param asb list of statement blocks
	 * @param C set of candidates
	 * @param sCount statement count
	 */
	private void rDetermineCandidates(ArrayList<StatementBlock> asb, HashSet<Candidate> C, Integer sCount) 
	{
		for(StatementBlock sb : asb ) // foreach statementblock in parforbody
			for( Statement s : sb._statements ) // foreach statement in statement block
			{
				sCount++;
				if( s instanceof ForStatement ) { //incl parfor
					//despite separate dependency analysis for each nested parfor, we need to 
					//recursively check nested parfor as well in order to ensure correcteness
					//of constantChecks with regard to outer indexes
					rDetermineCandidates(((ForStatement)s).getBody(), C, sCount);
				}
				else if( s instanceof WhileStatement ) {
					rDetermineCandidates(((WhileStatement)s).getBody(), C, sCount);
				}
				else if( s instanceof IfStatement ) {
					rDetermineCandidates(((IfStatement)s).getIfBody(), C, sCount);
					rDetermineCandidates(((IfStatement)s).getElseBody(), C, sCount);
				}
				else if( s instanceof FunctionStatement ) {
					rDetermineCandidates(((FunctionStatement)s).getBody(), C, sCount);
				}
				else if( s instanceof PrintStatement && ((PrintStatement)s).getType() == PRINTTYPE.STOP ) {
					raiseValidateError("PARFOR loop dependency analysis: " +
						"stop() statement is not allowed inside a parfor loop body.", false);
				}
				else if( s instanceof PrintStatement && ((PrintStatement)s).getType() == PRINTTYPE.ASSERT ) {
					raiseValidateError("PARFOR loop dependency analysis: " +
						"assert() statement is not allowed inside a parfor loop body.", false);
				}
				else {
					VariableSet vsUpdated = s.variablesUpdated();
					if( vsUpdated == null ) continue;
					for(String write : vsUpdated.getVariableNames()) {
						//add writes to non-local variables to candidate set
						if( !_vsParent.containsVariable(write) ) continue;
						List<DataIdentifier> dats = getDataIdentifiers( s, true );
						for( DataIdentifier dat : dats ) {
							boolean accum = (s instanceof AssignmentStatement
								&& ((AssignmentStatement)s).isAccumulator());
							C.add( new Candidate(write, dat, accum) );
						}
					}
				}
			}
	}

	/**
	 * This method recursively determines partitioning candidates for input variables. 
	 * Candidates are defined as index reads of non-local variables.
	 * 
	 * @param var variables
	 * @param asb list of statement blocks
	 * @param C list of partition formats
	 */
	private void rDeterminePartitioningCandidates(String var, ArrayList<StatementBlock> asb, List<PartitionFormat> C) 
	{
		for( StatementBlock sb : asb ) {
			if( sb instanceof FunctionStatementBlock ) {
				FunctionStatement fs = (FunctionStatement) sb.getStatement(0);
				rDeterminePartitioningCandidates(var, fs.getBody(), C);
			}
			else if( sb instanceof ForStatementBlock ) { //incl parfor
				ForStatementBlock fsb = (ForStatementBlock) sb;
				ForStatement fs = (ForStatement) fsb.getStatement(0);
				List<Hop> datsRead = new ArrayList<>();
				//predicate
				rGetDataIdentifiers(resetVisitStatus(fsb.getFromHops()), datsRead);
				rGetDataIdentifiers(resetVisitStatus(fsb.getToHops()), datsRead);
				rGetDataIdentifiers(resetVisitStatus(fsb.getIncrementHops()), datsRead);
				rDeterminePartitioningCandidates(var, datsRead, C);
				//for / parfor body
				rDeterminePartitioningCandidates(var, fs.getBody(), C);
			}
			else if( sb instanceof WhileStatementBlock ) {
				WhileStatementBlock wsb = (WhileStatementBlock) sb;
				WhileStatement ws = (WhileStatement) wsb.getStatement(0);
				List<Hop> datsRead = new ArrayList<>();
				//predicate
				rGetDataIdentifiers(resetVisitStatus(wsb.getPredicateHops()), datsRead);
				rDeterminePartitioningCandidates(var, datsRead, C);
				//while body
				rDeterminePartitioningCandidates(var, ws.getBody(), C);
			}
			else if( sb instanceof IfStatementBlock ) {
				IfStatementBlock isb = (IfStatementBlock) sb;
				IfStatement is = (IfStatement) isb.getStatement(0);
				List<Hop> datsRead = new ArrayList<>();
				//predicate
				rGetDataIdentifiers(resetVisitStatus(isb.getPredicateHops()), datsRead);
				rDeterminePartitioningCandidates(var, datsRead, C);
				//if and else branch
				rDeterminePartitioningCandidates(var, is.getIfBody(), C);
				rDeterminePartitioningCandidates(var, is.getElseBody(), C);
			}
			else if( sb.getHops() != null ) {
				Hop.resetVisitStatus(sb.getHops());
				List<Hop> datsRead = new ArrayList<>();
				for( Hop root : sb.getHops() )
					rGetDataIdentifiers(root, datsRead);
				rDeterminePartitioningCandidates(var, datsRead, C);
			}
		}
	}

	private void rDeterminePartitioningCandidates(String var, List<Hop> datsRead, List<PartitionFormat> C) {
		if( datsRead == null )
			return;
		for(Hop read : datsRead) {
			if( read instanceof IndexingOp && var.equals( read.getInput().get(0).getName() ) )
				C.add( determineAccessPattern((IndexingOp) read) );
			else if( HopRewriteUtils.isData(read, OpOpData.TRANSIENTREAD) && var.equals(read.getName()) )
				C.add( PartitionFormat.NONE );
		}
	}
	
	private static Hop resetVisitStatus(Hop hop) {
		return hop == null ? hop :
			hop.resetVisitStatus();
	}
	
	private PartitionFormat determineAccessPattern( IndexingOp rix ) {
		boolean isSpark = OptimizerUtils.isSparkExecutionMode();
		int blksz = ConfigurationManager.getBlocksize();
		PartitionFormat dpf = null;
		
		//1) get all bounds expressions for index access
		Hop rowL = rix.getInput().get(1);
		Hop rowU = rix.getInput().get(2);
		Hop colL = rix.getInput().get(3);
		Hop colU = rix.getInput().get(4);
		
		try {
			//2) decided on access pattern
			//COLUMN_WISE if all rows and access to single column
			if( rix.isAllRows() && colL == colU ) {
				dpf = PartitionFormat.COLUMN_WISE;
			}
			//ROW_WISE if all cols and access to single row
			else if( rix.isAllCols() && rowL == rowU ) {
				dpf = PartitionFormat.ROW_WISE;
			}
			//COLUMN_BLOCK_WISE
			else if( isSpark && rix.isAllRows() && colL != colU ) {
				LinearFunction l1 = getLinearFunction(colL, true);
				LinearFunction l2 = getLinearFunction(colU, true);
				dpf = !isAlignedBlocking(l1, l2, blksz) ? PartitionFormat.NONE :
					new PartitionFormat(PDataPartitionFormat.COLUMN_BLOCK_WISE_N, (int)l1._b[0]);
			}
			//ROW_BLOCK_WISE
			else if( isSpark && rix.isAllCols() && rowL != rowU ) {
				LinearFunction l1 = getLinearFunction(rowL, true);
				LinearFunction l2 = getLinearFunction(rowU, true);
				dpf = !isAlignedBlocking(l1, l2, blksz) ?  PartitionFormat.NONE :
					new PartitionFormat(PDataPartitionFormat.ROW_BLOCK_WISE_N, (int)l1._b[0]);
			}
			//NONE otherwise (conservative)
			else
				dpf = PartitionFormat.NONE;
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		return dpf;
	}
	
	private static boolean isAlignedBlocking(LinearFunction l1, LinearFunction l2, int blksz) {
		return (l1!=null && l2!=null && l1.equalSlope(l2) //same slope
			&& l1._b.length==1 && l1._b[0]<=blksz         //single index and block
			&& (l2._a - l1._a + 1 == l1._b[0])            //intercept difference is slope
			&& (blksz/l1._b[0])*l1._b[0] == blksz         //aligned slope
			&& l2.eval(1L) == l1._b[0] );                 //aligned intercept
	}
	
	private void rConsolidateResultVars(ArrayList<StatementBlock> asb, ArrayList<ResultVar> vars) 
	{
		for(StatementBlock sb : asb ) // foreach statementblock in parforbody
		{
			if( sb instanceof ParForStatementBlock )
				vars.addAll(((ParForStatementBlock)sb).getResultVariables());
			
			for( Statement s : sb._statements ) {
				if( s instanceof ForStatement || s instanceof ParForStatement )
					rConsolidateResultVars(((ForStatement)s).getBody(), vars);
				else if( s instanceof WhileStatement ) 
					rConsolidateResultVars(((WhileStatement)s).getBody(), vars);
				else if( s instanceof IfStatement ) {
					rConsolidateResultVars(((IfStatement)s).getIfBody(), vars);
					rConsolidateResultVars(((IfStatement)s).getElseBody(), vars);
				}
				else if( s instanceof FunctionStatement ) 
					rConsolidateResultVars(((FunctionStatement)s).getBody(), vars);
			}
		}
	}

	/**
	 * This method recursively checks a candidate against StatementBlocks for anti, data and output dependencies.
	 * A LanguageException is raised if at least one dependency is found, where it is guaranteed that no false negatives 
	 * (undetected dependency) but potentially false positives (misdetected dependency) can appear.  
	 * 
	 * 
	 * @param c candidate
	 * @param cdt candidate data type
	 * @param asb list of statement blocks
	 * @param sCount statement count
	 * @param dep array of boolean potential output dependencies
	 */
	private void rCheckCandidates(Candidate c, DataType cdt,
			ArrayList<StatementBlock> asb, Integer sCount, boolean[] dep) 
	{
		// check candidate only (output dependency if scalar or constant matrix subscript)
		if(    cdt == DataType.SCALAR 
			|| cdt == DataType.UNKNOWN  ) //dat2 checked for other candidate 
		{
			//every write to a scalar or complete data object is an output dependency
			dep[0] = true;
			if( ABORT_ON_FIRST_DEPENDENCY )
				return;
		}
		else if( cdt == DataType.MATRIX ) 
		{
			if( runConstantCheck(c._dat) && !c._isAccum ) {
				if( LOG.isTraceEnabled() )
					LOG.trace("PARFOR: Possible output dependency detected via constant self-check: var '"+c._var+"'.");
				dep[0] = true;
				if( ABORT_ON_FIRST_DEPENDENCY )
					return;
			}
		}
		
		// check candidate against all statements
		for(StatementBlock sb : asb )
			for( Statement s : sb._statements )
			{
				sCount++; 
				if( s instanceof ForStatement ) { //incl parfor
					//despite separate dependency analysis for each nested parfor, we need to 
					//recursively check nested parfor as well in order to ensure correcteness
					//of constantChecks with regard to outer indexes
					rCheckCandidates(c, cdt, ((ForStatement)s).getBody(), sCount, dep);
				}
				else if( s instanceof WhileStatement ) {
					rCheckCandidates(c, cdt, ((WhileStatement)s).getBody(), sCount, dep);
				}
				else if( s instanceof IfStatement ) {
					rCheckCandidates(c, cdt, ((IfStatement)s).getIfBody(), sCount, dep);
					rCheckCandidates(c, cdt, ((IfStatement)s).getElseBody(), sCount, dep);
				}
				else if( s instanceof FunctionStatement ) {
					rCheckCandidates(c, cdt, ((FunctionStatement)s).getBody(), sCount, dep);
				}
				else {
					//CHECK output dependencies
					List<DataIdentifier> datsUpdated = getDataIdentifiers(s, true);
					if( datsUpdated != null ) {
						for(DataIdentifier write : datsUpdated) {
							if( !c._var.equals( write.getName() ) ) continue;
							
							if( cdt != DataType.MATRIX && cdt != DataType.FRAME && cdt != DataType.LIST ) {
								//cannot infer type, need to exit (conservative approach)
								throw new LanguageException("PARFOR loop dependency analysis: cannot check "
									+ "for dependencies due to unknown datatype of var '"+c._var+"': "+cdt.name()+".");
							}
							
							DataIdentifier dat2 = write;
							if( c._dat == dat2 ) continue; //skip self-check
							if( runEqualsCheck(c._dat, dat2) ) {
								//intra-iteration output dependencies (same index function) are OK
							}
							else if(runBanerjeeGCDTest( c._dat, dat2 )) {
								LOG.trace("PARFOR: Possible output dependency detected via GCD/Banerjee: var '"+write+"'.");
								dep[0] = true;
								if( ABORT_ON_FIRST_DEPENDENCY )
									return;
							}
						}
					}
					
					List<DataIdentifier> datsRead = getDataIdentifiers(s, false);
					if( datsRead == null ) continue;
					
					//check data and anti dependencies
					for(DataIdentifier read : datsRead)
					{
						if( !c._var.equals( read.getName() ) ) continue;
						DataIdentifier dat2 = read;
						DataType dat2dt = _vsParent.getVariables().get(read.getName()).getDataType();
						
						if( cdt == DataType.SCALAR || cdt == DataType.UNKNOWN
							|| dat2dt == DataType.SCALAR || dat2dt == DataType.UNKNOWN )
						{
							//every write, read combination involving a scalar is a data dependency
							dep[1] = true;
							if( ABORT_ON_FIRST_DEPENDENCY )
								return;
						}
						else if( (cdt == DataType.MATRIX && dat2dt == DataType.MATRIX)
							|| (cdt == DataType.FRAME && dat2dt == DataType.FRAME )
							|| (cdt == DataType.LIST && dat2dt == DataType.LIST ) )
						{
							boolean invalid = false;
							if( runEqualsCheck(c._dat, dat2) )
								//read/write on same index, and not constant (checked for output) is OK
								invalid = runConstantCheck(dat2);
							else if( runBanerjeeGCDTest( c._dat, dat2 ) )
								invalid = true;
							else if( !(dat2 instanceof IndexedIdentifier) )
								//non-indexed access to candidate result variable -> always a dependency
								invalid = true;
							
							if( invalid ) {
								LOG.trace("PARFOR: Possible data/anti dependency detected via GCD/Banerjee: var '"+read+"'.");
								dep[1] = true;
								dep[2] = true;
								if( ABORT_ON_FIRST_DEPENDENCY )
									return;
							}
						}
						else { //if( c._dat.getDataType() == DataType.UNKNOWN )
							//cannot infer type, need to exit (conservative approach)
							throw new LanguageException("PARFOR loop dependency analysis: cannot check "
								+ "for dependencies due to unknown datatype of var '"+c._var+"': "+cdt.name()+".");
						}
					}
				}
			}
	}
	
	/**
	 * Get all target/source DataIdentifiers of the given statement.
	 * 
	 * @param s statement
	 * @param target if true, get targets
	 * @return list of data identifiers
	 */
	private List<DataIdentifier> getDataIdentifiers(Statement s, boolean target) 
	{
		List<DataIdentifier> ret = null;
		
		if( s instanceof AssignmentStatement ) {
			AssignmentStatement s2 = (AssignmentStatement)s;
			ret = target ? s2.getTargetList() :
				rGetDataIdentifiers(s2.getSource());
		}
		else if (s instanceof FunctionStatement) {
			FunctionStatement s2 = (FunctionStatement)s;
			ret = target ? s2.getOutputParams() :
				s2.getInputParams();
		}
		else if (s instanceof MultiAssignmentStatement) {
			MultiAssignmentStatement s2 = (MultiAssignmentStatement)s;
			ret = target ? s2.getTargetList() :
				rGetDataIdentifiers(s2.getSource());
		}
		else if (s instanceof PrintStatement) {
			PrintStatement s2 = (PrintStatement)s;
			ret = new ArrayList<>();
			for (Expression expression : s2.getExpressions())
				ret.addAll(rGetDataIdentifiers(expression));
		}
		
		//potentially extend this list with other Statements if required
		//(e.g., IOStatement, RandStatement)
		
		return ret;
	}

	private boolean isRowIgnorable(IndexedIdentifier dat1, IndexedIdentifier dat2) {
		for( IndexedIdentifier dat : new IndexedIdentifier[]{dat1,dat2} )
			if( !checkLower(dat1.getRowLowerBound(), dat.getRowLowerBound(), INTERAL_FN_INDEX_ROW)
				|| !checkLower(dat1.getRowUpperBound(), dat.getRowUpperBound(), INTERAL_FN_INDEX_ROW) )
				return false;
		return true;
	}
	
	private boolean isColumnIgnorable(IndexedIdentifier dat1, IndexedIdentifier dat2) {
		for( IndexedIdentifier dat : new IndexedIdentifier[]{dat1,dat2} )
			if( !checkLower(dat1.getColLowerBound(), dat.getColLowerBound(), INTERAL_FN_INDEX_COL)
				|| !checkLower(dat1.getColUpperBound(), dat.getColUpperBound(), INTERAL_FN_INDEX_COL) )
				return false;
		return true;
	}
	
	private boolean checkLower(Expression expr1, Expression expr2, String ix) {
		if( expr1 != null )
			for( DataIdentifier datsub : rGetDataIdentifiers(expr2) )
				if( _bounds._lower.containsKey(datsub.getName()) && !datsub.getName().startsWith(ix) )
					return false;
		return true;
	}
	
	private List<DataIdentifier> rGetDataIdentifiers(Expression e)
	{
		List<DataIdentifier> ret = new ArrayList<>();
		
		if( e instanceof DataIdentifier && !(e instanceof FunctionCallIdentifier
			|| e instanceof BuiltinFunctionExpression || e instanceof ParameterizedBuiltinFunctionExpression) ) {
			ret.add( (DataIdentifier)e );
		}
		else if( e instanceof FunctionCallIdentifier ) {
			FunctionCallIdentifier fci = (FunctionCallIdentifier)e;
			for( ParameterExpression ee : fci.getParamExprs() )
				ret.addAll(rGetDataIdentifiers( ee.getExpr() ));
		}
		else if(e instanceof BinaryExpression) {
			BinaryExpression be = (BinaryExpression) e;
			ret.addAll( rGetDataIdentifiers(be.getLeft()) );
			ret.addAll( rGetDataIdentifiers(be.getRight()) );
		}
		else if(e instanceof BooleanExpression) {
			BooleanExpression be = (BooleanExpression) e;
			ret.addAll( rGetDataIdentifiers(be.getLeft()) );
			ret.addAll( rGetDataIdentifiers(be.getRight()) );
		}
		else if(e instanceof BuiltinFunctionExpression) {
			BuiltinFunctionExpression be = (BuiltinFunctionExpression) e;
			//disregard meta data ops nrow/ncol (to exclude from candidates)
			if( !((be.getOpCode() == Builtins.NROW || be.getOpCode() == Builtins.NCOL)
				&& be.getFirstExpr() instanceof DataIdentifier) ) {
				ret.addAll( rGetDataIdentifiers(be.getFirstExpr()) );
				ret.addAll( rGetDataIdentifiers(be.getSecondExpr()) );
				ret.addAll( rGetDataIdentifiers(be.getThirdExpr()) );
			}
		}
		else if(e instanceof ParameterizedBuiltinFunctionExpression) {
			ParameterizedBuiltinFunctionExpression be = (ParameterizedBuiltinFunctionExpression) e;
			for( Expression ee : be.getVarParams().values() )
				ret.addAll( rGetDataIdentifiers(ee) );
		}
		else if(e instanceof RelationalExpression) {
			RelationalExpression re = (RelationalExpression) e;
			ret.addAll( rGetDataIdentifiers(re.getLeft()) );
			ret.addAll( rGetDataIdentifiers(re.getRight()) );
		}

		return ret;
	}
	
	private List<Hop> rGetDataIdentifiers(Hop root, List<Hop> direads) {
		if( root == null || root.isVisited() )
			return direads;
		//process children recursively (but disregard meta data ops and indexing)
		if( !((HopRewriteUtils.isUnary(root, OpOp1.NROW, OpOp1.NCOL)
			&& isDataIdentifier(root.getInput().get(0))) || isDataIdentifier(root)) ) {
			for( Hop c : root.getInput() )
				rGetDataIdentifiers(c, direads);
		}
		//handle transient read and right indexing over transient read
		if( isDataIdentifier(root) )
			direads.add(root);
		root.setVisited();
		return direads;
	}
	
	private static boolean isDataIdentifier(Hop hop) {
		return HopRewriteUtils.isData(hop, OpOpData.TRANSIENTREAD)
			|| (hop instanceof IndexingOp && HopRewriteUtils.isData(
			hop.getInput().get(0), OpOpData.TRANSIENTREAD))
			|| hop instanceof LiteralOp;
	}
	
	private void rDetermineBounds( ArrayList<StatementBlock> sbs, boolean flag ) {
		for( StatementBlock sb : sbs )
			rDetermineBounds(sb, flag);
	}
	
	/**
	 * Determines the lower/upper bounds of all nested for/parfor indexes.
	 * 
	 * @param sb statement block
	 * @param flag indicates that method is already in subtree of THIS.
	 */
	private void rDetermineBounds( StatementBlock sb, boolean flag ) 
	{
		// catch all known for/ parfor bounds 
		// (all unknown bounds are assumed to be +-infinity)
		
		for( Statement s : sb._statements )
		{
			boolean lFlag = flag;
			if( s instanceof ParForStatement || (s instanceof ForStatement && CONSERVATIVE_CHECK) ) //incl. for if conservative
			{
				ForStatement fs = (ForStatement)s;
				IterablePredicate ip = fs._predicate;
		
				//checks for position in overall tree
				if( sb==this )
					lFlag = true;
				
				if( lFlag || rIsParent(sb,this) ) //add only if in subtree of this
				{
					//check for internal names
					if(   ip.getIterVar()._name.equals( INTERAL_FN_INDEX_ROW )
					   || ip.getIterVar()._name.equals( INTERAL_FN_INDEX_COL ))
					{
						
						throw new LanguageException(" The iteration variable must not use the " +
								"internal iteration variable name prefix '"+ip.getIterVar()._name+"'.");
					}
					
					long low = Integer.MIN_VALUE;
					long up = Integer.MAX_VALUE;
					long incr = -1;
					
					if( ip.getFromExpr()instanceof IntIdentifier)
						low = ((IntIdentifier)ip.getFromExpr()).getValue();
					if( ip.getToExpr()instanceof IntIdentifier)
						up = ((IntIdentifier)ip.getToExpr()).getValue();
					
					//NOTE: conservative approach: include all index variables (also from for)
					if( ip.getIncrementExpr() instanceof IntIdentifier )
						incr = ((IntIdentifier)ip.getIncrementExpr()).getValue();
					else 
						incr = ( low <= up ) ? 1 : -1;
					
					//normalize bounds to positive increment (for dependency analysis only)
					if( incr < 0 ) {
						long tmp = low;
						low = up;
						up = tmp;
						incr *= -1; //positive increment
					}
					
					_bounds._lower.put(ip.getIterVar()._name, low);
					_bounds._upper.put(ip.getIterVar()._name, up);
					_bounds._increment.put(ip.getIterVar()._name, incr);
					if( lFlag ) //if local (required for constant check)
						_bounds._local.add(ip.getIterVar()._name);
				}	
				
				//recursive invocation (but not for nested parfors due to constant check)
				if( !lFlag )
					if( fs.getBody() != null )
						rDetermineBounds(fs.getBody(), lFlag);
			}
			else if( s instanceof ForStatement ) {
				ArrayList<StatementBlock> tmp = ((ForStatement) s).getBody();
				if( tmp != null )
					rDetermineBounds(tmp, lFlag);
			}
			else if( s instanceof WhileStatement ) {
				ArrayList<StatementBlock> tmp = ((WhileStatement) s).getBody();
				if( tmp != null )
					rDetermineBounds(tmp, lFlag);
			}
			else if( s instanceof IfStatement ) {
				ArrayList<StatementBlock> tmp = ((IfStatement) s).getIfBody();
				if( tmp != null )
					rDetermineBounds(tmp, lFlag);
				ArrayList<StatementBlock> tmp2 = ((IfStatement) s).getElseBody();
				if( tmp2 != null )
					rDetermineBounds(tmp2, lFlag);
			}
			else if( s instanceof FunctionStatement ) {
				ArrayList<StatementBlock> tmp = ((FunctionStatement) s).getBody();
				if( tmp != null )
					rDetermineBounds(tmp, lFlag);
			}
		}
	}
	
	private boolean rIsParent( ArrayList<StatementBlock> cParent, StatementBlock cChild) {
		return cParent.stream().anyMatch(sb -> rIsParent(sb, cChild));
	}

	private boolean rIsParent( StatementBlock cParent, StatementBlock cChild)
	{
		
		if( cParent == cChild )
			return true;
		
		boolean ret = false;
		for( Statement s : cParent.getStatements() ) {
			//check all the complex control flow constructs
			if( s instanceof ForStatement ) //for, parfor
				ret = rIsParent( ((ForStatement) s).getBody(), cChild );
			else if( s instanceof WhileStatement ) 
				ret = rIsParent( ((WhileStatement) s).getBody(), cChild );
			else if( s instanceof IfStatement ) {
				ret  = rIsParent( ((IfStatement) s).getIfBody(), cChild );
				ret |= rIsParent( ((IfStatement) s).getElseBody(), cChild );
			}
			
			//early return if already found
			if( ret ) break;
		}
		
		return ret;
	}

	/**
	 * Runs a combination of GCD and Banerjee test for a two potentially conflicting
	 * data identifiers. See below for a detailed explanation.
	 * 
	 * NOTE: simply enumerating all combinations of iteration variable values and probing for
	 * duplicates is not applicable due to (1) arbitrary nested program blocks with potentially
	 * dynamic lower, upper, and increment expressions, and (2) therefore potentially large 
	 * overheads in the general case.
	 * 
	 * @param dat1 data identifier 1
	 * @param dat2 data identifier 2
	 * @return true if "anti or data dependency"
	 */
	private boolean runBanerjeeGCDTest(DataIdentifier dat1, DataIdentifier dat2) 
	{
		/* The GCD (greatest common denominator) and the Banerjee test are two commonly used tests
		 * for determining loop-carried dependencies. Both rely on (1) linear index expressions of the
		 * form y = a + bx, where x is the loop index variable, and (2) conservative approaches that
		 * guarantee no false negatives (no missed dependencies) but possibly false positives. The GCD
		 * test probes for integer solutions without bounds, while the Banerjee test probes for real
		 * solutions with bounds. 
		 * 
		 * We use a combination of both:
		 * - the GCD test checks if dependencies are possible
		 * - the Banerjee test checks if those dependencies may arise within the given bounds
		 * 
		 * NOTES: 
		 * - #1 possible false positives may arise if there is a real solution within the bounds
		 * and an integer solution outside the bounds. This will lead to a detected dependencies
		 * although no integer solution within the bounds exists.
		 * - #2 for the sake of simplicity, we do not distinguish between anti and data dependencies,
		 * although possible in general
		 * - more advanced tests than GCD and Banerjee available (e.g., with symbolic checking for
		 *   non-linear functions) but this is a tradeoff between number of false positives and overhead
		 */
		
		LOG.trace("PARFOR: runBanerjeeGCDCheck.");
		
		boolean ret = true; //anti or data dependency
		
		//Step 1: analyze index expressions and transform them into linear functions
		LinearFunction f1 = getLinearFunction(dat1); 
		LinearFunction f2 = getLinearFunction(dat2);
		forceConsistency(f1,f2);
		
		LOG.trace("PARFOR: f1: " + f1.toString());
		LOG.trace("PARFOR: f2: " + f2.toString());
		
		///////
		//Step 2: run GCD Test 
		///////
		long lgcd = f1._b[0];
		for( int i=1; i<f1._b.length; i++ )
			lgcd = determineGCD( lgcd, f1._b[i] );
		for( int i=0; i<f2._b.length; i++ )
			lgcd = determineGCD( lgcd, f2._b[i] );
		
		if( (Math.abs(f1._a-f2._a) % lgcd) != 0 ) { //if GCD divides the intercepts
			//no integer solution exists -> no dependency
			ret = false;
		}
		
		LOG.trace("PARFOR: GCD result: "+ret);

		if( !CONSERVATIVE_CHECK && ret ) //only if not already no dependency
		{
			//NOTE: cases both and none negligible already covered (constant check, general case) 
			boolean ixid = (dat1 instanceof IndexedIdentifier && dat2 instanceof IndexedIdentifier); 
			boolean ignoreRow = ixid && isRowIgnorable((IndexedIdentifier)dat1, (IndexedIdentifier)dat2);
			boolean ignoreCol = ixid && isColumnIgnorable((IndexedIdentifier)dat1, (IndexedIdentifier)dat2);
	
			LinearFunction f1p = null, f2p = null;
			if( ignoreRow ) {
				f1p = getColLinearFunction(dat1);
				f2p = getColLinearFunction(dat2);
			}
			if( ignoreCol ) {
				f1p = getRowLinearFunction(dat1);
				f2p = getRowLinearFunction(dat2);
			}
			
			LOG.trace("PARFOR: f1p: "+((f1p==null)?"null":f1p.toString()));
			LOG.trace("PARFOR: f2p: "+((f2p==null)?"null":f2p.toString()));
			
			if( f1p!=null && f2p!=null )
			{
				forceConsistency(f1p, f2p);
				
				long lgcd2 = f1p._b[0];
				for( int i=1; i<f1p._b.length; i++ )
					lgcd2 = determineGCD( lgcd2, f1p._b[i] );
				for( int i=0; i<f2p._b.length; i++ )
					lgcd2 = determineGCD( lgcd2, f2p._b[i] );
				
				if( (Math.abs(f1p._a-f2p._a) % lgcd2) != 0 ) { //if GCD divides the intercepts
					//no integer solution exists -> no dependency
					ret = false;
				}
				
				LOG.trace("PARFOR: GCD result: "+ret);
			}
		}
		
		
		///////
		//Step 3: run Banerjee Test
		///////
		if( ret ) //only if GCD found possible dependencies
		{
			//determining anti/data dependencies
			long lintercept = f2._a - f1._a;
			long lmax=0;
			long lmin=0;

			//min/max bound 
			int len = Math.max(f1._b.length, f2._b.length);
			boolean invalid = false;
			for( int i=0; i<len; i++ )
			{
				String var=(f1._b.length>i) ? f1._vars[i] : f2._vars[i];
				if( !_bounds._lower.containsKey(var) || !_bounds._upper.containsKey(var) ) {
					invalid = true; break;
				}
				
				//get lower and upper bound for specific var or internal var
				long lower = _bounds._lower.get(var); //bounds equal for f1 and f2
				long upper = _bounds._upper.get(var);
				
				//max bound
				if( f1._b.length>i )
					lmax += (f1._b[i]>0) ? f1._b[i]*upper : f1._b[i]*lower;
				if( f2._b.length>i )
					lmax -= (f2._b[i]>0) ? f2._b[i]*lower : f2._b[i]*upper; 
				
				//min bound (unequal indexes)
				if( f1._b.length>i )
					lmin += (f1._b[i]>0) ? f1._b[i]*lower : f1._b[i]*upper;
				if( f2._b.length>i )
					lmin -= (f2._b[i]>0) ? f2._b[i]*upper : f2._b[i]*lower;
			}

			if( LOG.isTraceEnabled() )
				LOG.trace("PARFOR: Banerjee lintercept=" + lintercept+", lmax="+lmax+", lmin="+lmin+", invalid="+invalid);
			
			if( !invalid && (!(lmin <= lintercept && lintercept <= lmax) || lmin==lmax) ) {
				//dependency not within the bounds of the arrays
				ret = false;
			}
			
			LOG.trace("PARFOR: Banerjee result: "+ret);
		}
	
		return ret;
	}

	/**
	 * Runs a constant check for a single data identifier (target of assignment). If constant, then every
	 * iteration writes to the same cell. 
	 * 
	 * @param dat1 data identifier
	 * @return true if dependency
	 */
	private boolean runConstantCheck(DataIdentifier dat1) 
	{
		LOG.trace("PARFOR: runConstantCheck.");
		
		boolean ret = true; //data dependency to itself
		LinearFunction f1 = getLinearFunction(dat1);
		if( f1 == null )
			return true; //dependency 
		
		LOG.trace("PARFOR: f1: "+f1.toString());
		
		// no output dependency to itself if no index access will happen twice
		// hence we check for: (all surrounding indexes are used by f1 and all intercepts != 0 )
		boolean gcheck=true;
		for( String var : _bounds._local ) //check only local, nested checked from parent
		{
			if(   var.startsWith(INTERAL_FN_INDEX_ROW) 
			   || var.startsWith(INTERAL_FN_INDEX_COL)) 
			{
				continue; //skip internal vars for range indexing 
			}
			
			boolean lcheck = false;
			for( int i=0; i<f1._vars.length; i++ )
				if( var.equals(f1._vars[i]) )
					if( f1._b[i] != 0 )
						lcheck = true;
			if( !lcheck )
			{
				gcheck=false;
				break;
			}
		}
		
		if( gcheck ) // output dependencies impossible
			ret = false;
		
		return ret;
	}
	
	/**
	 * Runs an equality check for two data identifiers. If equal, there there are no
	 * inter-iteration (loop-carried) but only intra-iteration dependencies.
	 * 
	 * @param dat1 data identifier 1
	 * @param dat2 data identifier 2
	 * @return true if equal data identifiers
	 */
	private boolean runEqualsCheck(DataIdentifier dat1, DataIdentifier dat2) 
	{
		LOG.trace("PARFOR: runEqualsCheck.");
		
		//check if both data identifiers of same type
		if(dat1 instanceof IndexedIdentifier != dat2 instanceof IndexedIdentifier)
			return false;
			
		//general case function comparison
		boolean ret = true; //true if equal index functions
		LinearFunction f1 = getLinearFunction(dat1);
		LinearFunction f2 = getLinearFunction(dat2);
		forceConsistency(f1, f2);
		ret = f1.equals(f2);
		
		LOG.trace("PARFOR: f1: " + f1.toString());
		LOG.trace("PARFOR: f2: " + f2.toString());
		LOG.trace("PARFOR: (f1==f2): " + ret);
		
		//additional check if cols/rows could be ignored
		if( !CONSERVATIVE_CHECK && !ret ) //only if not already equal
		{
			//NOTE: cases both and none negligible already covered (constant check, general case) 
			boolean ixid = (dat1 instanceof IndexedIdentifier && dat2 instanceof IndexedIdentifier); 
			boolean ignoreRow = ixid && isRowIgnorable((IndexedIdentifier)dat1, (IndexedIdentifier)dat2);
			boolean ignoreCol = ixid && isColumnIgnorable((IndexedIdentifier)dat1, (IndexedIdentifier)dat2);
	
			LinearFunction f1p = null, f2p = null;
			if( ignoreRow ) {
				f1p = getColLinearFunction(dat1);
				f2p = getColLinearFunction(dat2);
			}
			if( ignoreCol ) {
				f1p = getRowLinearFunction(dat1);
				f2p = getRowLinearFunction(dat2);
			}
			
			if( f1p!=null && f2p!=null ) {
				forceConsistency(f1p, f2p);
				ret = f1p.equals(f2p);
				
				LOG.trace("PARFOR: f1p: " + f1p.toString());
				LOG.trace("PARFOR: f2p: " + f2p.toString());
				LOG.trace("PARFOR: (f1p==f2p): " + ret);
			}
		}
		
		return ret;
	}
	
	/**
	 * This is the Euclid's algorithm for GCD (greatest common denominator), 
	 * required for the GCD test.
	 * 
	 * @param a first value
	 * @param b second value
	 * @return greatest common denominator
	 */
	private long determineGCD(long a, long b) {
		return (b==0) ? a : determineGCD(b, a % b);
	}

	/**
	 * Creates or reuses a linear function for a given data identifier, where identifiers with equal
	 * names and matrix subscripts result in exactly the same linear function.
	 * 
	 * @param dat data identifier
	 * @return linear function
	 */
	private LinearFunction getLinearFunction(DataIdentifier dat)
	{
		/* Notes:
		 * - Currently, this function supports 2dim matrix subscripts with arbitrary linear functions
		 *   however, this could be extended to d-dim if necessary
		 * - Trick for range indexing: introduce a pseudo index variable with lower and upper according to 
		 *   the index range (e.g., [1:4,...]) or matrix dimensionality (e.g., [:,...]). This allows us to
		 *   apply existing tests even for range indexing (multi-value instead of single-value functions)
		 */

		LinearFunction out = null;
		
		if( ! (dat instanceof IndexedIdentifier ) ) //happens if matrix is now used as scalar
			return new LinearFunction(0,0,dat.getName());
		
		IndexedIdentifier idat = (IndexedIdentifier) dat;
		
		if( USE_FN_CACHE ) {
			out = _fncache.get( getFunctionID(idat) );
			if( out != null ) 
				return out; 
		}
		
		Expression sub1 = idat.getRowLowerBound();
		Expression sub2 = idat.getColLowerBound();
		
		//parse row expressions
		try
		{
			//loop index or constant (default case)
			if( idat.getRowLowerBound()!=null && idat.getRowUpperBound()!=null &&
					idat.getRowLowerBound() == idat.getRowUpperBound()         ) 
			{
				if( sub1 instanceof IntIdentifier )
					out = new LinearFunction(((IntIdentifier)sub1).getValue(), 0, null);
				else if( sub1 instanceof DataIdentifier )
					out = new LinearFunction(0, 1, ((DataIdentifier)sub1)._name);
				else
					out = rParseBinaryExpression((BinaryExpression)sub1);
				
				if( !CONSERVATIVE_CHECK )
					if(out.hasNonIndexVariables())
					{
						String id = INTERAL_FN_INDEX_ROW+_idSeqfn.getNextID();
						out = new LinearFunction(0, 1L, id);
						
						_bounds._lower.put(id, 1L);
						_bounds._upper.put(id, _vsParent.getVariable(idat._name).getDim1()); //row dim
						_bounds._increment.put(id, 1L);
					}
			}
			else //range indexing
			{
				Expression sub1a = sub1;
				Expression sub1b = idat.getRowUpperBound();
				
				String id = INTERAL_FN_INDEX_ROW+_idSeqfn.getNextID();
				out = new LinearFunction(0, 1L, id);
				
				if( sub1a == null && sub1b == null //: operator
					|| !(sub1a instanceof IntIdentifier) || !(sub1b instanceof IntIdentifier) ) { //for robustness
					_bounds._lower.put(id, 1L);
					_bounds._upper.put(id, _vsParent.getVariable(idat._name).getDim1()); //row dim
					_bounds._increment.put(id, 1L);
				}
				else if( sub1a instanceof IntIdentifier && sub1b instanceof IntIdentifier ) {
					_bounds._lower.put(id, ((IntIdentifier)sub1a).getValue());
					_bounds._upper.put(id, ((IntIdentifier)sub1b).getValue()); 
					_bounds._increment.put(id, 1L);
				}
				else {
					out = null;
				}
			}
			
			//scale row function 'out' with col dimensionality	
			long colDim = _vsParent.getVariable(idat._name).getDim2();
			if( colDim >= 0 ) {
				out.scale( colDim );
			}
			else {
				//NOTE: we could mark sb for deferred validation and evaluate on execute (see ParForProgramBlock)
				LOG.debug("PARFOR: Warning - matrix dimensionality of '"+idat._name+"' unknown, cannot scale linear functions.");				
			}
		}
		catch(Exception ex) {
			LOG.debug("PARFOR: Unable to parse MATRIX subscript expression for '"+String.valueOf(sub1)+"'.", ex);
			out = null; //let dependency analysis fail
		}
		
		//parse col expression and merge functions
		if( out!=null )
		{
			try
			{
				LinearFunction tmpOut = null;
				
				//loop index or constant (default case)
				if( idat.getColLowerBound()!=null && idat.getColUpperBound()!=null &&
						idat.getColLowerBound() == idat.getColUpperBound()             ) 
				{
					if( sub2 instanceof IntIdentifier )
						out.addConstant( ((IntIdentifier)sub2).getValue() );
					else if( sub2 instanceof DataIdentifier )
						tmpOut = new LinearFunction(0, 1, ((DataIdentifier)sub2)._name) ;
					else
						tmpOut = rParseBinaryExpression((BinaryExpression)sub2);
					
					if( !CONSERVATIVE_CHECK )
						if(tmpOut!=null && tmpOut.hasNonIndexVariables())
						{
							String id = INTERAL_FN_INDEX_COL+_idSeqfn.getNextID();
							tmpOut = new LinearFunction(0, 1L, id); 
							_bounds._lower.put(id, 1l);
							_bounds._upper.put(id, _vsParent.getVariable(idat._name).getDim2()); //col dim
							_bounds._increment.put(id, 1L);	
						}
				}
				else //range indexing
				{
					Expression sub2a = sub2;
					Expression sub2b = idat.getColUpperBound();
					
					String id = INTERAL_FN_INDEX_COL+_idSeqfn.getNextID();
					tmpOut = new LinearFunction(0, 1L, id);
					
					if(   sub2a == null && sub2b == null  //: operator 
					   || !(sub2a instanceof IntIdentifier) || !(sub2b instanceof IntIdentifier) ) //for robustness
					{
						_bounds._lower.put(id, 1L);
						_bounds._upper.put(id, _vsParent.getVariable(idat._name).getDim2()); //col dim
						_bounds._increment.put(id, 1L);					
					}
					else if( sub2a instanceof IntIdentifier && sub2b instanceof IntIdentifier )
					{
						_bounds._lower.put(id, ((IntIdentifier)sub2a).getValue());
						_bounds._upper.put(id, ((IntIdentifier)sub2b).getValue()); 
						_bounds._increment.put(id, 1L);
					}
					else
					{
						out = null;
					}
				}
				
				//final merge of row and col functions
				if( tmpOut != null )
					out.addFunction(tmpOut);
			}
			catch(Exception ex)
			{
				LOG.debug("PARFOR: Unable to parse MATRIX subscript expression for '"+String.valueOf(sub2)+"'.", ex);
				out = null; //let dependency analysis fail
			}
		}
		
		//post processing after creation
		if( out != null )
		{
			//cleanup and verify created function; raise exceptions if needed
			cleanupFunction(out);
			verifyFunction(out);
			
			// pseudo loop normalization of functions (incr=1, from=1 not necessary due to Banerjee) 
			// (precondition for GCD test)
			if( NORMALIZE ) {
				int index=0;
				for( String var : out._vars ) {
					long low  = _bounds._lower.get(var);
					long up   = _bounds._upper.get(var);
					long incr = _bounds._increment.get(var);
					if( incr < 0 || 1 < incr ) { //does never apply to internal (artificial) vars
						out.normalize(index,low,incr); // normalize linear functions
						_bounds._upper.put(var,(long)Math.ceil(((double)up)/incr)); // normalize upper bound
					}
					index++;
				}
			}
			
			//put into cache
			if( USE_FN_CACHE )
				_fncache.put( getFunctionID(idat), out );
		}
		
		return out;
	}
	
	private LinearFunction getRowLinearFunction(DataIdentifier dat) 
	{
		//NOTE: would require separate function cache, not realized due to inexpensive operations
		
		LinearFunction out = null;
		IndexedIdentifier idat = (IndexedIdentifier) dat;
		Expression sub1 = idat.getRowLowerBound();
		
		try
		{
			//loop index or constant (default case)
			if( idat.getRowLowerBound()!=null && idat.getRowUpperBound()!=null &&
					idat.getRowLowerBound() == idat.getRowUpperBound()         ) 
			{
				if( sub1 instanceof IntIdentifier )
					out = new LinearFunction(((IntIdentifier)sub1).getValue(), 0, null);
				else if( sub1 instanceof DataIdentifier )
					out = new LinearFunction(0, 1, ((DataIdentifier)sub1).getName());
				else
					out = rParseBinaryExpression((BinaryExpression)sub1);
			}
		}
		catch(Exception ex) {
			LOG.debug("PARFOR: Unable to parse MATRIX subscript expression for '"+String.valueOf(sub1)+"'.", ex);
			out = null; //let dependency analysis fail
		}
		
		//post processing after creation
		if( out != null ) {
			//cleanup and verify created function; raise exceptions if needed
			cleanupFunction(out);
			verifyFunction(out);
		}
		
		return out;
	}
	
	private LinearFunction getColLinearFunction(DataIdentifier dat) 
	{
		//NOTE: would require separate function cache, not realized due to inexpensive operations
		
		LinearFunction out = null;
		IndexedIdentifier idat = (IndexedIdentifier) dat;
		Expression sub1 = idat.getColLowerBound();
		
		try
		{
			//loop index or constant (default case)
			if( idat.getColLowerBound()!=null && idat.getColUpperBound()!=null &&
					idat.getColLowerBound() == idat.getColUpperBound()         ) 
			{
				if( sub1 instanceof IntIdentifier )
					out = new LinearFunction(((IntIdentifier)sub1).getValue(), 0, null);
				else if( sub1 instanceof DataIdentifier )
					out = new LinearFunction(0, 1, ((DataIdentifier)sub1).getName());
				else
					out = rParseBinaryExpression((BinaryExpression)sub1);
			}
		}
		catch(Exception ex) {
			LOG.debug("PARFOR: Unable to parse MATRIX subscript expression for '"+String.valueOf(sub1)+"'.", ex);
			out = null; //let dependency analysis fail
		}
		
		//post processing after creation
		if( out != null ) {
			//cleanup and verify created function; raise exceptions if needed
			cleanupFunction(out);
			verifyFunction(out);
		}
		
		return out;
	}
	
	@SuppressWarnings("unused")
	private LinearFunction getLinearFunction(Expression expr, boolean ignoreMinWithConstant) {
		if( expr instanceof IntIdentifier )
			return new LinearFunction(((IntIdentifier)expr).getValue(), 0, null);
		else if( expr instanceof BinaryExpression )
			return rParseBinaryExpression((BinaryExpression)expr);
		else if( expr instanceof BuiltinFunctionExpression && ignoreMinWithConstant ) {
			//note: builtin function expression is also a data identifier and hence order before
			BuiltinFunctionExpression bexpr = (BuiltinFunctionExpression) expr;
			if( bexpr.getOpCode()==Builtins.MIN ) {
				if( bexpr.getFirstExpr() instanceof BinaryExpression )
					return rParseBinaryExpression((BinaryExpression)bexpr.getFirstExpr());
				else if( bexpr.getSecondExpr() instanceof BinaryExpression )
					return rParseBinaryExpression((BinaryExpression)bexpr.getSecondExpr());
			}
		}
		else if( expr instanceof DataIdentifier )
			return new LinearFunction(0, 1, ((DataIdentifier)expr).getName());
		
		return null;
	}
	
	private LinearFunction getLinearFunction(Hop hop, boolean ignoreMinWithConstant) {
		if( hop instanceof LiteralOp && hop.getValueType()==ValueType.INT64 )
			return new LinearFunction(HopRewriteUtils.getIntValue((LiteralOp)hop), 0, null);
		else if( HopRewriteUtils.isBinary(hop, OpOp2.PLUS, OpOp2.MINUS, OpOp2.MULT) )
			return rParseBinaryExpression(hop);
		else if( HopRewriteUtils.isBinary(hop, OpOp2.MIN) && ignoreMinWithConstant ) {
			//note: builtin function expression is also a data identifier and hence order before
			if( hop.getInput().get(0) instanceof org.apache.sysds.hops.BinaryOp )
				return rParseBinaryExpression(hop.getInput().get(0));
			else if( hop.getInput().get(1) instanceof org.apache.sysds.hops.BinaryOp )
				return rParseBinaryExpression(hop.getInput().get(1));
		}
		else if( HopRewriteUtils.isData(hop, OpOpData.TRANSIENTREAD) )
			return new LinearFunction(0, 1, hop.getName());
		
		return null;
	}
	
	/**
	 * Creates a functionID for a given data identifier (mainly used for caching purposes),
	 * where data identifiers with equal name and matrix subscripts results in equal
	 * functionIDs.
	 * 
	 * @param dat indexed identifier
	 * @return string function id
	 */
	private static String getFunctionID( IndexedIdentifier dat )
	{
		// note: using dat.hashCode can be different for same functions, 
		// hence, we use a custom String ID
		IndexedIdentifier idat = dat;
		Expression ex1a = idat.getRowLowerBound();
		Expression ex1b = idat.getRowUpperBound();
		Expression ex2a = idat.getColLowerBound();
		Expression ex2b = idat.getColUpperBound();
		
		StringBuilder sb = new StringBuilder();
		sb.append(String.valueOf(ex1a));
		sb.append(',');
		sb.append(String.valueOf(ex1b));
		sb.append(',');
		sb.append(String.valueOf(ex2a));
		sb.append(',');
		sb.append(String.valueOf(ex2b));
		return sb.toString();
	}
	
	
	
	/**
	 * Removes all zero intercepts created by recursive computation.
	 * 
	 * @param f1 linear function
	 */
	private static void cleanupFunction( LinearFunction f1 ) {
		for( int i=0; i<f1._b.length; i++ )
			if( f1._vars[i]==null ) {
				f1.removeVar(i);
				i--; 
				continue;
			}
	}
	
	/**
	 * Simply verification check of created linear functions, mainly used for
	 * robustness purposes.
	 * 
	 * @param f1 linear function
	 */
	private void verifyFunction(LinearFunction f1)
	{
		//check for required form of linear functions
		if( f1 == null || f1._b.length != f1._vars.length ) {
			if( LOG.isTraceEnabled() && f1!=null ) 
				LOG.trace("PARFOR: f1: "+f1.toString());
			throw new LanguageException("PARFOR loop dependency analysis: " +
				"MATRIX subscripts are not in linear form (a0 + a1*x).");
		}
		
		//check all function variables to be index variables
		for( String var : f1._vars )
		{
			if( !_bounds._lower.containsKey(var) ) {
				LOG.trace("PARFOR: not allowed variable in matrix subscript: "+var);
				throw new LanguageException("PARFOR loop dependency analysis: " +
					"MATRIX subscripts use non-index variables."); 
			}
		}
	}
	
	/**
	 * Tries to obtain consistent linear functions by forcing the same variable ordering for
	 * efficient comparison: f2 is modified in a way that it matches the sequence of variables in f1.
	 * 
	 * @param f1 linear function 1
	 * @param f2 linear function 2
	 */
	private static void forceConsistency(LinearFunction f1, LinearFunction f2) 
	{
		boolean warn = false;

		for( int i=0; i<f1._b.length; i++ )
		{
			if( f2._b.length<(i+1) )
				break;
			
			if(   !f1._vars[i].equals(f2._vars[i])
			    &&!(f1._vars[i].startsWith(INTERAL_FN_INDEX_ROW) && f2._vars[i].startsWith(INTERAL_FN_INDEX_ROW)) 
			    &&!(f1._vars[i].startsWith(INTERAL_FN_INDEX_COL) && f2._vars[i].startsWith(INTERAL_FN_INDEX_COL)))
			{
				boolean exchange = false;
				//scan 
				for( int j=i+1; j<f2._b.length; j++ )
					if(    f1._vars[i].equals(f2._vars[j]) 
						||(f1._vars[i].startsWith(INTERAL_FN_INDEX_ROW) && f2._vars[j].startsWith(INTERAL_FN_INDEX_ROW)) 
						||(f1._vars[i].startsWith(INTERAL_FN_INDEX_COL) && f2._vars[j].startsWith(INTERAL_FN_INDEX_COL)) )
					{
						//exchange
						long btmp = f2._b[i];
						String vartmp = f2._vars[i];
						f2._b[i] = f2._b[j];
						f2._vars[i] = f2._vars[j];
						f2._b[j] = btmp;
						f2._vars[j] = vartmp;
						exchange = true;
					}
				if( !exchange )
					warn = true;
			}
		}

		
		if( warn && LOG.isTraceEnabled() )
			LOG.trace( "PARFOR: Warning - index functions f1 and f2 cannot be made consistent." );
	}
	
	/**
	 * Recursively creates a linear function for a single BinaryExpression, where PLUS, MINUS, MULT
	 * are allowed as operators.
	 * 
	 * @param be binary expression
	 * @return linear function
	 */
	private LinearFunction rParseBinaryExpression(BinaryExpression be) {
		Expression l = be.getLeft();
		Expression r = be.getRight();
		if( be.getOpCode() == BinaryOp.PLUS || be.getOpCode() == BinaryOp.MINUS ) {
			boolean plus = be.getOpCode() == BinaryOp.PLUS;
			//parse binary expressions
			if( l instanceof BinaryExpression) {
				LinearFunction f = rParseBinaryExpression((BinaryExpression) l);
				Long cvalR = parseLongConstant(r);
				if( f != null && cvalR != null )
					return f.addConstant(cvalR * (plus?1:-1));
			}
			else if (r instanceof BinaryExpression) {
				LinearFunction f = rParseBinaryExpression((BinaryExpression) r);
				Long cvalL = parseLongConstant(l);
				if( f != null && cvalL != null )
					return f.scale(plus?1:-1).addConstant(cvalL);
			}
			else { // atomic case
				//change everything to plus if necessary
				Long cvalL = parseLongConstant(l);
				Long cvalR = parseLongConstant(r);
				if( cvalL != null )
					return new LinearFunction(cvalL,plus?1:-1,((DataIdentifier)r)._name);
				else if( cvalR != null )
					return new LinearFunction(cvalR*(plus?1:-1),1,((DataIdentifier)l)._name);
			}
		}
		else if( be.getOpCode() == BinaryOp.MULT ) {
			//atomic case (only recursion for MULT expressions, where one side is a constant)
			Long cvalL = parseLongConstant(l);
			Long cvalR = parseLongConstant(r);
			if( cvalL != null && r instanceof DataIdentifier )
				return new LinearFunction(0, cvalL,((DataIdentifier)r)._name);
			else if( cvalR != null && l instanceof DataIdentifier )
				return new LinearFunction(0, cvalR,((DataIdentifier)l)._name);
			else if( cvalL != null && r instanceof BinaryExpression )
				return rParseBinaryExpression((BinaryExpression)r).scale(cvalL);
			else if( cvalR != null && l instanceof BinaryExpression )
				return rParseBinaryExpression((BinaryExpression)l).scale(cvalR);
		}
		return null; //let dependency analysis fail
	}
	
	private LinearFunction rParseBinaryExpression(Hop hop) {
		org.apache.sysds.hops.BinaryOp bop = (org.apache.sysds.hops.BinaryOp) hop;
		Hop l = bop.getInput().get(0);
		Hop r = bop.getInput().get(1);
		if( bop.getOp()==OpOp2.PLUS || bop.getOp()==OpOp2.MINUS ) {
			boolean plus = bop.getOp() == OpOp2.PLUS;
			//parse binary expressions
			if( l instanceof org.apache.sysds.hops.BinaryOp) {
				LinearFunction f = rParseBinaryExpression(l);
				Long cvalR = parseLongConstant(r);
				if( f != null && cvalR != null )
					return f.addConstant(cvalR * (plus?1:-1));
			}
			else if (r instanceof org.apache.sysds.hops.BinaryOp) {
				LinearFunction f = rParseBinaryExpression(r);
				Long cvalL = parseLongConstant(l);
				if( f != null && cvalL != null )
					return f.scale(plus?1:-1).addConstant(cvalL);
			}
			else { // atomic case
				//change everything to plus if necessary
				Long cvalL = parseLongConstant(l);
				Long cvalR = parseLongConstant(r);
				if( cvalL != null )
					return new LinearFunction(cvalL, plus?1:-1, r.getName() );
				else if( cvalR != null )
					return new LinearFunction(cvalR*(plus?1:-1), 1, l.getName());
			}
		}
		else if( bop.getOp() == OpOp2.MULT ) {
			//atomic case (only recursion for MULT expressions, where one side is a constant)
			Long cvalL = parseLongConstant(l);
			Long cvalR = parseLongConstant(r);
			if( cvalL != null && HopRewriteUtils.isData(r, OpOpData.TRANSIENTREAD) )
				return new LinearFunction(0, cvalL, r.getName());
			else if( cvalR != null && HopRewriteUtils.isData(l, OpOpData.TRANSIENTREAD) )
				return new LinearFunction(0, cvalR, l.getName());
			else if( cvalL != null && r instanceof org.apache.sysds.hops.BinaryOp )
				return rParseBinaryExpression(r).scale(cvalL);
			else if( cvalR != null && l instanceof org.apache.sysds.hops.BinaryOp )
				return rParseBinaryExpression(l).scale(cvalR);
		}
		return null; //let dependency analysis fail
	}

	private static Long parseLongConstant(Expression expr) {
		if( expr instanceof IntIdentifier ) {
			return ((IntIdentifier) expr).getValue();
		}
		else if( expr instanceof DoubleIdentifier ) {
			double tmp = ((DoubleIdentifier) expr).getValue();
			if( tmp == Math.floor(tmp) ) //ensure int
				return UtilFunctions.toLong(tmp);
		}
		return null;
	}
	
	private static Long parseLongConstant(Hop hop) {
		if( hop instanceof LiteralOp && hop.getValueType()==ValueType.INT64 ) {
			return HopRewriteUtils.getIntValue((LiteralOp)hop);
		}
		else if( hop instanceof LiteralOp && hop.getValueType()==ValueType.FP64 ) {
			double tmp = HopRewriteUtils.getDoubleValue((LiteralOp)hop);
			if( tmp == Math.floor(tmp) ) //ensure int
				return UtilFunctions.toLong(tmp);
		}
		return null;
	}
	
	public static class ResultVar {
		public final String _name;
		public final boolean _isAccum;
		public ResultVar(String name, boolean accum) {
			_name = name;
			_isAccum = accum;
		}
		@Override
		public boolean equals(Object that) {
			String varname = (that instanceof ResultVar) ?
				((ResultVar)that)._name : that.toString();
			return _name.equals(varname);
		}
		@Override
		public int hashCode() {
			return _name.hashCode();
		}
		@Override
		public String toString() {
			return _name;
		}
		public static boolean contains(Collection<ResultVar> list, String varName) {
			//helper function which is necessary because list.contains checks
			//varName.equals(rvar) which always returns false because it not a string
			return list.stream().anyMatch(rvar -> rvar._name.equals(varName));
		}
	}
	
	private static class Candidate  {
		private final String _var;          // variable name
		private final DataIdentifier _dat;  // _var data identifier
		private final boolean _isAccum;
		public Candidate(String var, DataIdentifier di, boolean accum) {
			_var = var;
			_dat = di;
			_isAccum = accum;
		}
	}
	
	/**
	 * Helper class for representing all lower, upper bounds of (potentially nested)
	 * loop constructs. 
	 *
	 */
	private static class Bounds {
		HashMap<String, Long> _lower     = new HashMap<>();
		HashMap<String, Long> _upper     = new HashMap<>();
		HashMap<String, Long> _increment = new HashMap<>();
		//contains all local variable names (subset of lower/upper/incr sets)
		HashSet<String> _local = new HashSet<>();
	}
	
	/**
	 * Helper class for representing linear functions of matrix subscripts.
	 * The allowed form is 'y = a + b1x1 + ... = bnxn', which is required by
	 * the applied GCD and Banerjee tests.
	 *
	 */
	private class LinearFunction {
		long _a;        // intercept
		long[] _b;      // slopes 
		String[] _vars; // b variable names
		
		LinearFunction( long a, long b, String name ) {
			_a       = a;
			_b       = new long[1];
			_b[0]    = b;
			_vars    = new String[1];
			_vars[0] = name;
		}
		
		public LinearFunction addConstant(long value) {
			_a += value;
			return this;
		}

		public LinearFunction addFunction( LinearFunction f2) {
			_a = _a + f2._a;
			long[] tmpb = new long[_b.length+f2._b.length];
			System.arraycopy( _b,    0, tmpb, 0,         _b.length    );
			System.arraycopy( f2._b, 0, tmpb, _b.length, f2._b.length );
			_b = tmpb;
			String[] tmpvars = new String[_vars.length+f2._vars.length];
			System.arraycopy( _vars,    0, tmpvars, 0,            _vars.length    );
			System.arraycopy( f2._vars, 0, tmpvars, _vars.length, f2._vars.length );
			_vars = tmpvars;
			return this;
		}

		public LinearFunction removeVar( int i ) {
			long[] tmpb = new long[_b.length-1];
			System.arraycopy( _b, 0, tmpb, 0, i );
			System.arraycopy( _b, i+1, tmpb, i, _b.length-i-1 );
			_b = tmpb;
			String[] tmpvars = new String[_vars.length-1];
			System.arraycopy( _vars, 0, tmpvars, 0, i );
			System.arraycopy( _vars, i+1, tmpvars, i, _vars.length-i-1 );
			_vars = tmpvars;
			return this;
		}
		
		public LinearFunction scale( long scale ) {
			_a *= scale; 
			for( int i=0; i<_b.length; i++ )
				_b[i] *= scale;
			return this;
		}
		
		public LinearFunction normalize(int index, long lower, long increment) {
			_a -= (_b[index] * lower);
			_b[index] *= increment;
			return this;
		}
		
		public long eval(Long... x) {
			long ret = _a;
			for( int i=0; i<_b.length; i++ )
				ret += _b[i] *= x[i];
			return ret;
		}
		
		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append("(");
			sb.append(_a);
			sb.append(") + ");
			sb.append("(");			
			for( int i=0; i<_b.length; i++ ) {
				if( i>0 )
					sb.append("+");
				sb.append("(");
				sb.append(_b[i]);
				sb.append(" * ");
				sb.append(_vars[i]);
				sb.append(")");
			}
			sb.append(")");
			return sb.toString();
		}
		
		@Override
		public boolean equals( Object o2 ) {
			if( o2 == null || !(o2 instanceof LinearFunction)  )
				return false;
			LinearFunction f2 = (LinearFunction)o2;
			return ( _a == f2._a )
				&& equalSlope(f2);
		}

		public boolean equalSlope(LinearFunction f2) {
			boolean ret = ( _b.length == f2._b.length );
			for( int i=0; i<_b.length && ret; i++ ) {
				ret &= (_b[i] == f2._b[i] );
				//note robustness for null var names 
				String var1 = String.valueOf(_vars[i]);
				String var2 = String.valueOf(f2._vars[i]);
				ret &= (var1.equals(var2)
					||(var1.startsWith(INTERAL_FN_INDEX_ROW) && var2.startsWith(INTERAL_FN_INDEX_ROW))
					||(var1.startsWith(INTERAL_FN_INDEX_COL) && var2.startsWith(INTERAL_FN_INDEX_COL)));
			}
			return ret;
		}
		
		@Override
		public int hashCode() {
			return super.hashCode(); //identity
		}
		
		public boolean hasNonIndexVariables() {
			for( String var : _vars )
				if( var!=null && !_bounds._lower.containsKey(var) )
					return true;
			return false;
		}
	}
}
