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

package org.apache.sysds.hops.ipa;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;
import java.util.stream.Collectors;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;

public class FunctionCallGraph 
{
	//internal function key for main program (underscore 
	//prevents any conflicts with user-defined functions)
	private static final String MAIN_FUNCTION_KEY = "_main"; 
	
	//unrolled function call graph, in call direction
	//(mapping from function keys to called function keys)
	private final Map<String, Set<String>> _fGraph;
	
	//program-wide function call operators per target function
	//(mapping from function keys to set of its function calls)
	private final Map<String, List<FunctionOp>> _fCalls;
	private final Map<String, List<StatementBlock>> _fCallsSB;
	
	//subset of direct or indirect recursive functions
	private final Set<String> _fRecursive;

	//subset of side-effect-free functions
	private final Set<String> _fSideEffectFree;
	
	// a boolean value to indicate if exists the second order function (e.g. eval, paramserv)
	// and the UDFs that are marked secondorder="true"
	private final boolean _containsSecondOrder;
	
	/**
	 * Constructs the function call graph for all functions
	 * reachable from the main program. 
	 * 
	 * @param prog dml program of given script
	 */
	public FunctionCallGraph(DMLProgram prog) {
		_fGraph = new HashMap<>();
		_fCalls = new HashMap<>();
		_fCallsSB = new HashMap<>();
		_fRecursive = new HashSet<>();
		_fSideEffectFree = new HashSet<>();
		_containsSecondOrder = constructFunctionCallGraph(prog);
	}
	
	/**
	 * Constructs the function call graph for all functions
	 * reachable from the given statement block. 
	 * 
	 * @param sb statement block (potentially hierarchical)
	 */
	public FunctionCallGraph(StatementBlock sb) {
		_fGraph = new HashMap<>();
		_fCalls = new HashMap<>();
		_fCallsSB = new HashMap<>();
		_fRecursive = new HashSet<>();
		_fSideEffectFree = new HashSet<>();
		_containsSecondOrder = constructFunctionCallGraph(sb);
	}

	/**
	 * Returns all functions called from the given function. 
	 * 
	 * @param fnamespace function namespace
	 * @param fname function name
	 * @return set of function keys (namespace and name)
	 */
	public Set<String> getCalledFunctions(String fnamespace, String fname) {
		return getCalledFunctions(
			DMLProgram.constructFunctionKey(fnamespace, fname));
	}
	
	/**
	 * Returns all functions called from the given function. 
	 * 
	 * @param fkey function key of calling function, null indicates the main program
	 * @return set of function keys (namespace and name)
	 */
	public Set<String> getCalledFunctions(String fkey) {
		String lfkey = (fkey == null) ? MAIN_FUNCTION_KEY : fkey;
		return _fGraph.get(lfkey);
	}
	
	/**
	 * Returns all function operators calling the given function.
	 * 
	 * @param fkey function key of called function,
	 *      null indicates the main program and returns an empty list
	 * @return list of function call hops 
	 */
	public List<FunctionOp> getFunctionCalls(String fkey) {
		//main program cannot have function calls
		if( fkey == null )
			return Collections.emptyList();
		return _fCalls.get(fkey);
	}
	
	/**
	 * Returns all statement blocks that contain a function operator
	 * calling the given function.
	 * 
	 * @param fkey function key of called function,
	 *      null indicates the main program and returns an empty list
	 * @return list of statement blocks
	 */
	public List<StatementBlock> getFunctionCallsSB(String fkey) {
		//main program cannot have function calls
		if( fkey == null )
			return Collections.emptyList();
		return _fCallsSB.get(fkey);
	}
	
	/**
	 * Removes all calls of the given function.
	 * 
	 * @param fkey function key of called function,
	 *      null indicates the main program, which has no affect
	 */
	public void removeFunctionCalls(String fkey) {
		_fCalls.remove(fkey);
		_fCallsSB.remove(fkey);
		_fRecursive.remove(fkey);
		_fGraph.remove(fkey);
		for( Entry<String, Set<String>> e : _fGraph.entrySet() )
			e.getValue().removeIf(s -> s.equals(fkey));
	}
	
	/**
	 * Removes a single function call identified by target function name,
	 * and source function op and statement block.
	 * 
	 * @param fkey function key of called function
	 * @param fop source function call operator 
	 * @param sb source statement block
	 */
	public void removeFunctionCall(String fkey, FunctionOp fop, StatementBlock sb) {
		if( _fCalls.containsKey(fkey) )
			_fCalls.get(fkey).remove(fop);
		if( _fCallsSB.containsKey(fkey) )
			_fCallsSB.get(fkey).remove(sb);
	}
	
	/**
	 * Replaces a function call to fkeyOld with a call to fkey,
	 * but using the function op and statement block from the old.
	 * 
	 * @param fkeyOld old function key of called function
	 * @param fkey new function key of called function
	 */
	public void replaceFunctionCalls(String fkeyOld, String fkey) {
		List<FunctionOp> fopTmp = _fCalls.get(fkeyOld);
		List<StatementBlock> sbTmp =_fCallsSB.get(fkeyOld);
		_fCalls.remove(fkeyOld);
		_fCallsSB.remove(fkeyOld);
		_fCalls.put(fkey, fopTmp);
		_fCallsSB.put(fkey, sbTmp);
		//additional cleanups fold no longer reachable
		_fRecursive.remove(fkeyOld);
		_fSideEffectFree.remove(fkeyOld);
		_fGraph.remove(fkeyOld);
		for( Set<String> hs : _fGraph.values() )
			hs.remove(fkeyOld);
	}
	
	/**
	 * Indicates if the given function is either directly or indirectly recursive.
	 * An example of an indirect recursive function is foo2 in the following call
	 * chain: foo1 -&gt; foo2 -&gt; foo1.
	 * 
	 * @param fnamespace function namespace
	 * @param fname function name
	 * @return true if the given function is recursive, false otherwise
	 */
	public boolean isRecursiveFunction(String fnamespace, String fname) {
		return isRecursiveFunction(
			DMLProgram.constructFunctionKey(fnamespace, fname));
	}
	
	/**
	 * Indicates if the given function is either directly or indirectly recursive.
	 * An example of an indirect recursive function is foo2 in the following call
	 * chain: foo1 -&gt; foo2 -&gt; foo1.
	 * 
	 * @param fkey function key of calling function, null indicates the main program
	 * @return true if the given function is recursive, false otherwise
	 */
	public boolean isRecursiveFunction(String fkey) {
		String lfkey = (fkey == null) ? MAIN_FUNCTION_KEY : fkey;
		return _fRecursive.contains(lfkey);
	}
	
	/**
	 * Indicates if the given function is side effect free, i.e., has no
	 * prints, no persistent write, and includes no or only calls to
	 * side-effect-free functions.
	 * 
	 * @param fnamespace function namespace
	 * @param fname function name
	 * @return true if the given function is side-effect-free, false otherwise
	 */
	public boolean isSideEffectFreeFunction(String fnamespace, String fname) {
		return isSideEffectFreeFunction(
			DMLProgram.constructFunctionKey(fnamespace, fname));
	}
	
	/**
	 * Indicates if the given function is side effect free, i.e., has no
	 * prints, no persistent write, and includes no or only calls to
	 * side-effect-free functions.
	 * 
	 * @param fkey function key of calling function, null indicates the main program
	 * @return true if the given function is side-effect-free, false otherwise
	 */
	public boolean isSideEffectFreeFunction(String fkey) {
		String lfkey = (fkey == null) ? MAIN_FUNCTION_KEY : fkey;
		return _fSideEffectFree.contains(lfkey);
	}
	
	/**
	 * Returns all functions that are reachable either directly or indirectly
	 * form the main program, except the main program itself.
	 * 
	 * @return set of function keys (namespace and name)
	 */
	public Set<String> getReachableFunctions() {
		return getReachableFunctions(Collections.emptySet());
	}
	
	/**
	 * Returns all functions that are reachable either directly or indirectly
	 * form the main program, except the main program itself and the given 
	 * exclude-list of function names.
	 * 
	 * @param excludeList list of function keys to exclude
	 * @return set of function keys (namespace and name)
	 */
	public Set<String> getReachableFunctions(Set<String> excludeList) {
		return _fGraph.keySet().stream()
			.filter(p -> !excludeList.contains(p) && !MAIN_FUNCTION_KEY.equals(p))
			.collect(Collectors.toSet());
	}
	
	/**
	 * Indicates if the given function is reachable either directly or indirectly
	 * from the main program.
	 * 
	 * @param fnamespace function namespace
	 * @param fname function name
	 * @return true if the given function is reachable, false otherwise
	 */
	public boolean isReachableFunction(String fnamespace, String fname) {
		return isReachableFunction(
			DMLProgram.constructFunctionKey(fnamespace, fname));
	}
	
	/**
	 * Indicates if the given function is reachable either directly or indirectly
	 * from the main program.
	 * 
	 * @param fkey function key of calling function, null indicates the main program
	 * @return true if the given function is reachable, false otherwise
	 */
	public boolean isReachableFunction(String fkey) {
		return isReachableFunction(fkey, false);
	}
	
	/**
	 * Indicates if the given function is reachable either directly or indirectly
	 * from the main program.
	 * 
	 * @param fkey function key of calling function, null indicates the main program
	 * @param deep if all reachability lists need to be probed (no short cuts)
	 * @return true if the given function is reachable, false otherwise
	 */
	protected boolean isReachableFunction(String fkey, boolean deep) {
		//we check only entry points as functions removed if no longer reachable,
		//otherwise, we check all reachability lists deeply
		String lfkey = (fkey == null) ? MAIN_FUNCTION_KEY : fkey;
		return !deep ? _fGraph.containsKey(lfkey) : _fGraph.values()
			.stream().anyMatch(list -> list.contains(lfkey));
	}
	
	/**
	 * Indicates if the function call graph, i.e., functions that are transitively
	 * reachable from the main program, contains a second-order builtin function call 
	 * (e.g., eval, paramserv), which prohibits the removal of unused functions.
	 *
	 * @return true if the function call graph contains a second-order builtin function call.
	 */
	public boolean containsSecondOrderCall() {
		return _containsSecondOrder;
	}
	
	private boolean constructFunctionCallGraph(DMLProgram prog) {
		if( !prog.hasFunctionStatementBlocks() ) {
			boolean ret = false;
			for( StatementBlock sb : prog.getStatementBlocks() )
				ret |= rAnalyzeSecondOrderCall(sb);
			return ret; //early abort if prog without functions
		}
		
		boolean ret = false;
		try {
			//construct the main function call graph
			Stack<String> fstack = new Stack<>();
			Set<String> lfset = new HashSet<>();
			_fGraph.put(MAIN_FUNCTION_KEY, new HashSet<String>());
			for( StatementBlock sblk : prog.getStatementBlocks() )
				ret |= rConstructFunctionCallGraph(MAIN_FUNCTION_KEY, sblk, fstack, lfset);
			
			//analyze all non-recursive functions if free of side effects
			_fSideEffectFree.addAll(_fCalls.keySet().stream()
				.filter(s -> !s.startsWith(DMLProgram.INTERNAL_NAMESPACE))
				.filter(s -> isSideEffectFree(prog.getFunctionStatementBlock(s)))
				.collect(Collectors.toList()));
		}
		catch(HopsException ex) {
			throw new RuntimeException(ex);
		}
		return ret;
	}
	
	private boolean constructFunctionCallGraph(StatementBlock sb) {
		if( !sb.getDMLProg().hasFunctionStatementBlocks() )
			return false; //early abort if prog without functions
		
		try {
			Stack<String> fstack = new Stack<>();
			Set<String> lfset = new HashSet<>();
			_fGraph.put(MAIN_FUNCTION_KEY, new HashSet<String>());
			return rConstructFunctionCallGraph(MAIN_FUNCTION_KEY, sb, fstack, lfset);
		}
		catch(HopsException ex) {
			throw new RuntimeException(ex);
		}
	}
	
	private boolean rConstructFunctionCallGraph(String fkey, StatementBlock sb, Stack<String> fstack, Set<String> lfset) {
		boolean ret = false;
		if (sb instanceof WhileStatementBlock) {
			WhileStatement ws = (WhileStatement)sb.getStatement(0);
			for (StatementBlock current : ws.getBody())
				ret |= rConstructFunctionCallGraph(fkey, current, fstack, lfset);
		} 
		else if (sb instanceof IfStatementBlock) {
			IfStatement ifs = (IfStatement) sb.getStatement(0);
			for (StatementBlock current : ifs.getIfBody())
				ret |= rConstructFunctionCallGraph(fkey, current, fstack, lfset);
			for (StatementBlock current : ifs.getElseBody())
				ret |= rConstructFunctionCallGraph(fkey, current, fstack, lfset);
		} 
		else if (sb instanceof ForStatementBlock) {
			ForStatement fs = (ForStatement)sb.getStatement(0);
			for (StatementBlock current : fs.getBody())
				ret |= rConstructFunctionCallGraph(fkey, current, fstack, lfset);
		} 
		//FunctionStatementBlock handled on adding functions from basic blocks
		else {
			// For generic StatementBlock
			List<Hop> hopsDAG = sb.getHops();
			if( hopsDAG == null || hopsDAG.isEmpty() ) 
				return false; //nothing to do

			ret = HopRewriteUtils.containsSecondOrderBuiltin(hopsDAG);
			Hop.resetVisitStatus(hopsDAG);
			for( Hop h : hopsDAG ) {
				//function ops can only occur as root nodes of the dag
				if( h instanceof FunctionOp ) {
					ret |= addFunctionOpToGraph((FunctionOp) h, fkey, sb, fstack, lfset);
				}
				
				//recursive processing for paramserv functions
				rConstructFunctionCallGraph(h, fkey, sb, fstack, lfset);
			}
		}
		
		return ret;
	}
	
	private boolean rConstructFunctionCallGraph(Hop hop, String fkey, StatementBlock sb, Stack<String> fstack, Set<String> lfset) {
		boolean ret = false;
		if( hop.isVisited() )
			return ret;
		
		//recursively process all child nodes
		for( Hop h : hop.getInput() )
			rConstructFunctionCallGraph(h, fkey, sb, fstack, lfset);
		
		if( HopRewriteUtils.isParameterizedBuiltinOp(hop, ParamBuiltinOp.PARAMSERV)
			&& HopRewriteUtils.knownParamservFunctions(hop, sb.getDMLProg()))
		{
			ParameterizedBuiltinOp pop = (ParameterizedBuiltinOp) hop;
			List<FunctionOp> fps = pop.getParamservPseudoFunctionCalls();
			//include artificial function ops into functional call graph
			if( !fps.isEmpty() ) //valid functional parameters
				for( FunctionOp fop : fps )
					ret |= addFunctionOpToGraph(fop, fkey, sb, fstack, lfset);
		}
		
		hop.setVisited();
		return ret;
	}
	
	private boolean addFunctionOpToGraph(FunctionOp fop, String fkey, StatementBlock sb, Stack<String> fstack, Set<String> lfset) {
		try{
			boolean ret = false;
			String lfkey = fop.getFunctionKey();
			//keep all function operators
			if( !_fCalls.containsKey(lfkey) ) {
				_fCalls.put(lfkey, new ArrayList<>());
				_fCallsSB.put(lfkey, new ArrayList<>());
			}
			_fCalls.get(lfkey).add(fop);
			_fCallsSB.get(lfkey).add(sb);

			//prevent redundant call edges
			if( lfset.contains(lfkey) || fop.getFunctionNamespace().equals(DMLProgram.INTERNAL_NAMESPACE) )
				return ret;

			if( !_fGraph.containsKey(lfkey) )
				_fGraph.put(lfkey, new HashSet<String>());

			//recursively construct function call dag
			if( !fstack.contains(lfkey) ) {
				fstack.push(lfkey);
				_fGraph.get(fkey).add(lfkey);
				FunctionStatementBlock fsb = sb.getDMLProg()
					.getFunctionStatementBlock(fop.getFunctionNamespace(), fop.getFunctionName());
				FunctionStatement fs = (FunctionStatement) fsb.getStatement(0);
				for( StatementBlock csb : fs.getBody() )
					ret |= rConstructFunctionCallGraph(lfkey, csb, fstack, new HashSet<String>());
				fstack.pop();
			}
			//recursive function call
			else {
				_fGraph.get(fkey).add(lfkey);
				_fRecursive.add(lfkey);
			
				//mark indirectly recursive functions as recursive
				int ix = fstack.indexOf(lfkey);
				for( int i=ix+1; i<fstack.size(); i++ )
					_fRecursive.add(fstack.get(i));
			}

			//mark as visited for current function call context
			lfset.add( lfkey );
			return ret;
		}
		catch(Exception e){
			throw new DMLException("failed add function to graph " + fop + " " + fkey , e );
		}
	}

	private boolean rAnalyzeSecondOrderCall(StatementBlock sb) {
		boolean ret = false;
		if (sb instanceof WhileStatementBlock) {
			WhileStatement ws = (WhileStatement)sb.getStatement(0);
			for (StatementBlock current : ws.getBody())
				ret |= rAnalyzeSecondOrderCall(current);
		}
		else if (sb instanceof IfStatementBlock) {
			IfStatement ifs = (IfStatement) sb.getStatement(0);
			for (StatementBlock current : ifs.getIfBody())
				ret |= rAnalyzeSecondOrderCall(current);
			for (StatementBlock current : ifs.getElseBody())
				ret |= rAnalyzeSecondOrderCall(current);
		}
		else if (sb instanceof ForStatementBlock) {
			ForStatement fs = (ForStatement)sb.getStatement(0);
			for (StatementBlock current : fs.getBody())
				ret |= rAnalyzeSecondOrderCall(current);
		}
		else {
			// For generic StatementBlock
			List<Hop> hopsDAG = sb.getHops();
			if( hopsDAG == null || hopsDAG.isEmpty() ) 
				return false; //nothing to do
			//function ops can only occur as root nodes of the dag
			ret = HopRewriteUtils.containsSecondOrderBuiltin(hopsDAG);
		}
		return ret;
	}
	
	private static boolean isSideEffectFree(FunctionStatementBlock fsb) {
		//check regular dml-bodied function for prints, pwrite, and other functions
		FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
		for( StatementBlock csb : fstmt.getBody() )
			if( rHasSideEffects(csb) )
				return false;
		return true;
	}
	
	private static boolean rHasSideEffects(StatementBlock sb) {
		boolean ret = false;
		if( sb instanceof ForStatementBlock ) {
			ForStatement fstmt = (ForStatement) sb.getStatement(0);
			for( StatementBlock csb : fstmt.getBody() )
				ret |= rHasSideEffects(csb);
		}
		else if( sb instanceof WhileStatementBlock ) {
			WhileStatement wstmt = (WhileStatement) sb.getStatement(0);
			for( StatementBlock csb : wstmt.getBody() )
				ret |= rHasSideEffects(csb);
		}
		else if( sb instanceof IfStatementBlock ) {
			IfStatement istmt = (IfStatement) sb.getStatement(0);
			for( StatementBlock csb : istmt.getIfBody() )
				ret |= rHasSideEffects(csb);
			if( istmt.getElseBody() != null )
				for( StatementBlock csb : istmt.getElseBody() )
					ret |= rHasSideEffects(csb);
		}
		else if( sb.getHops() != null ) {
			//check for print, printf, pwrite, function calls, all of
			//which can only appear as root nodes in the DAG
			for( Hop root : sb.getHops() ) {
				ret |= HopRewriteUtils.isUnary(root, OpOp1.PRINT)
					|| HopRewriteUtils.isNary(root, OpOpN.PRINTF)
					|| HopRewriteUtils.isData(root, OpOpData.PERSISTENTWRITE)
					|| root instanceof FunctionOp;
			}
		}
		return ret;
	}
}
