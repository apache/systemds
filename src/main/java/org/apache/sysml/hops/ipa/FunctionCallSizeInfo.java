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

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;

/**
 * Auxiliary data structure to hold function call summaries in terms
 * of information about number of function calls, consistent dimensions,
 * consistent sparsity, and dimension-preserving functions.
 * 
 */
public class FunctionCallSizeInfo 
{
	//basic function call graph to obtain size information
	private final FunctionCallGraph _fgraph;
	
	//functions that are subject to size propagation
	//(called once or multiple times with consistent sizes)
	private final Set<String> _fcand;
	
	//functions that are not subject to size propagation
	//but preserve the dimensions (used to propagate inputs
	//to subsequent statement blocks and functions)
	private final Set<String> _fcandUnary;
	
	//indicators for which function arguments of valid functions it 
	//is safe to propagate the number of non-zeros 
	//(mapping from function keys to set of function input positions)
	private final Map<String, Set<Integer>> _fcandSafeNNZ;
	
	//indicators which literal function arguments can be safely 
	//propagated into and replaced in the respective functions 
	//(mapping from function keys to set of function input positions)
	private final Map<String, Set<Integer>> _fSafeLiterals;
	
	/**
	 * Constructs the function call summary for all functions
	 * reachable from the main program. 
	 * 
	 * @param fgraph function call graph
	 * @throws HopsException 
	 */
	public FunctionCallSizeInfo(FunctionCallGraph fgraph) 
		throws HopsException 
	{
		this(fgraph, true);
	}
	
	/**
	 * Constructs the function call summary for all functions
	 * reachable from the main program. 
	 * 
	 * @param fgraph function call graph
	 * @param init initialize function candidates
	 * @throws HopsException 
	 */
	public FunctionCallSizeInfo(FunctionCallGraph fgraph, boolean init) 
		throws HopsException 
	{
		_fgraph = fgraph;
		_fcand = new HashSet<String>();
		_fcandUnary = new HashSet<String>();
		_fcandSafeNNZ =  new HashMap<String, Set<Integer>>();
		_fSafeLiterals = new HashMap<String, Set<Integer>>();
		
		constructFunctionCallSizeInfo();
	}
	
	/**
	 * Gets the number of function calls to a given function.
	 * 
	 * @param fkey function key
	 * @return number of function calls
	 */
	public int getFunctionCallCount(String fkey) {
		return _fgraph.getFunctionCalls(fkey).size();
	}
	
	/**
	 * Indicates if the given function is valid for statistics
	 * propagation.
	 * 
	 * @param fkey function key
	 * @return true if valid
	 */
	public boolean isValidFunction(String fkey) {
		return _fcand.contains(fkey);
	}
	
	/**
	 * Gets the set of functions that are valid for statistics
	 * propagation.
	 * 
	 * @return set of function keys
	 */
	public Set<String> getValidFunctions() {
		return _fcand;
	}
	
	/**
	 * Gets the set of functions that are invalid for statistics
	 * propagation. This is literally the set of reachable
	 * functions minus the set of valid functions.
	 * 
	 * @return set of function keys.
	 */
	public Set<String> getInvalidFunctions() {
		return _fgraph.getReachableFunctions(getValidFunctions());
	}
	
	/**
	 * Adds a function to the set of dimension-preserving
	 * functions.
	 * 
	 * @param fkey function key
	 */
	public void addDimsPreservingFunction(String fkey) {
		_fcandUnary.add(fkey);
	}
	
	/**
	 * Gets the set of dimension-preserving functions, i.e., 
	 * functions with one matrix input and output of equal
	 * dimension sizes.
	 * 
	 * @return set of function keys
	 */
	public Set<String> getDimsPreservingFunctions() {
		return _fcandUnary;
	}
	
	/**
	 * Indicates if the given function belongs to the set
	 * of dimension-preserving functions. 
	 * 
	 * @param fkey function key
	 * @return true if the function is dimension-preserving
	 */
	public boolean isDimsPreservingFunction(String fkey) {
		return _fcandUnary.contains(fkey);
	}
	
	/**
	 * Indicates if the given function input allows for safe 
	 * nnz propagation, i.e., all function calls have a consistent 
	 * number of non-zeros.  
	 * 
	 * @param fkey function key
	 * @param pos function input position
	 * @return true if nnz can safely be propagated
	 */
	public boolean isSafeNnz(String fkey, int pos) {
		return _fcandSafeNNZ.containsKey(fkey)
			&& _fcandSafeNNZ.get(fkey).contains(pos);
	}
	
	/**
	 * Indicates if the given function has at least one input
	 * that allows for safe literal propagation and replacement,
	 * i.e., all function calls have consistent literal inputs.
	 * 
	 * @param fkey function key
	 * @return true if a literal can be safely propagated
	 */
	public boolean hasSafeLiterals(String fkey) {
		return _fSafeLiterals.containsKey(fkey)
			&& !_fSafeLiterals.get(fkey).isEmpty();
	}
	
	/**
	 * Indicates if the given function input allows for safe
	 * literal propagation and replacement, i.e., all function calls
	 * have consistent literal inputs.
	 * 
	 * @param fkey function key
	 * @param pos function input position
	 * @return true if literal that can be safely propagated
	 */
	public boolean isSafeLiteral(String fkey, int pos) {
		return _fSafeLiterals.containsKey(fkey)
			&& _fSafeLiterals.get(fkey).contains(pos);
	}
	
	private void constructFunctionCallSizeInfo() 
		throws HopsException 
	{
		//step 1: determine function candidates by evaluating all function calls
		for( String fkey : _fgraph.getReachableFunctions() ) {
			List<FunctionOp> flist = _fgraph.getFunctionCalls(fkey);
		
			//condition 1: function called just once
			if( flist.size() == 1 ) {
				_fcand.add(fkey);
			}
			//condition 2: check for consistent input sizes
			else if( InterProceduralAnalysis.ALLOW_MULTIPLE_FUNCTION_CALLS ) {
				//compare input matrix characteristics of first against all other calls
				FunctionOp first = flist.get(0);
				boolean consistent = true;
				for( int i=1; i<flist.size(); i++ ) {
					FunctionOp other = flist.get(i);
					for( int j=0; j<first.getInput().size(); j++ ) {
						Hop h1 = first.getInput().get(j);
						Hop h2 = other.getInput().get(j);
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
				}
				if( consistent )
					_fcand.add(fkey);
			}
		}
		
		//step 2: determine safe nnz propagation per input
		//(considered for valid functions only)
		for( String fkey : _fcand ) {
			FunctionOp first = _fgraph.getFunctionCalls(fkey).get(0);
			HashSet<Integer> tmp = new HashSet<Integer>();
			for( int j=0; j<first.getInput().size(); j++ ) {
				//if nnz known it is safe to propagate those nnz because for multiple calls 
				//we checked of equivalence and hence all calls have the same nnz
				Hop input = first.getInput().get(0);
				if( input.getNnz()>=0 ) 
					tmp.add(j);
			}
			_fcandSafeNNZ.put(fkey, tmp);
		}
		
		//step 3: determine safe literal replacement per function input
		//(considered for all functions)
		for( String fkey : _fgraph.getReachableFunctions() ) {
			List<FunctionOp> flist = _fgraph.getFunctionCalls(fkey);
			FunctionOp first = flist.get(0);
			//initialize w/ all literals of first call
			HashSet<Integer> tmp = new HashSet<Integer>();
			for( int j=0; j<first.getInput().size(); j++ )
				if( first.getInput().get(j) instanceof LiteralOp )
					tmp.add(j);
			//check consistency across all function calls
			for( int i=1; i<flist.size(); i++ ) {
				FunctionOp other = flist.get(i);
				for( int j=0; j<first.getInput().size(); j++ ) 
					if( tmp.contains(j) ) {
						Hop h1 = first.getInput().get(j);
						Hop h2 = other.getInput().get(j);
						if( !(h2 instanceof LiteralOp && HopRewriteUtils
							.isEqualValue((LiteralOp)h1, (LiteralOp)h2)) )
							tmp.remove(j);
					}
			}
			_fSafeLiterals.put(fkey, tmp);
		}
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		sb.append("Valid functions for propagation: \n");
		for( String fkey : getValidFunctions() ) {
			sb.append("--");
			sb.append(fkey);
			sb.append(": ");
			sb.append(getFunctionCallCount(fkey));
			if( !_fcandSafeNNZ.get(fkey).isEmpty() ) {
				sb.append("\n----");
				sb.append(Arrays.toString(_fcandSafeNNZ.get(fkey).toArray(new Integer[0])));
			}
			sb.append("\n");
		}
		
		if( !getInvalidFunctions().isEmpty() ) {
			sb.append("Invaid functions for propagation: \n");
			for( String fkey : getInvalidFunctions() ) {
				sb.append("--");
				sb.append(fkey);
				sb.append(": ");
				sb.append(getFunctionCallCount(fkey));
				sb.append("\n");
			}
		}
		
		if( !getDimsPreservingFunctions().isEmpty() ) {
			sb.append("Dimensions-preserving functions: \n");
			for( String fkey : getDimsPreservingFunctions() ) {
				sb.append("--");
				sb.append(fkey);
				sb.append(": ");
				sb.append(getFunctionCallCount(fkey));
				sb.append("\n");
			}
		}
		
		sb.append("Valid scalars for propagation: \n");
		for( Entry<String, Set<Integer>> e : _fSafeLiterals.entrySet() ) {
			sb.append("--");
			sb.append(e.getKey());
			sb.append(": ");
			for( Integer pos : e.getValue() ) {
				sb.append(pos);
				sb.append(":");
				sb.append(_fgraph.getFunctionCalls(e.getKey())
					.get(0).getInput().get(pos).getName());
				sb.append(" ");
			}
			sb.append("\n");
		}
		
		sb.append("Valid #non-zeros for propagation: \n");
		for( Entry<String, Set<Integer>> e : _fcandSafeNNZ.entrySet() ) {
			sb.append("--");
			sb.append(e.getKey());
			sb.append(": ");
			for( Integer pos : e.getValue() ) {
				sb.append(pos);
				sb.append(":");
				sb.append(_fgraph.getFunctionCalls(e.getKey())
					.get(0).getInput().get(pos).getName());
				sb.append(" ");
			}
			sb.append("\n");
		}
		
		return sb.toString();
	}
}
