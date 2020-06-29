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

package org.apache.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;

/**
 * Rule: Constant Folding. For all statement blocks, 
 * eliminate simple binary expressions of literals within dags by 
 * computing them and replacing them with a new Literal op once.
 * For the moment, this only applies within a dag, later this should be 
 * extended across statements block (global, inter-procedure). 
 */
public class RewriteConstantFolding extends HopRewriteRule
{
	private static final String TMP_VARNAME = "__cf_tmp";
	
	//reuse basic execution runtime
	private BasicProgramBlock _tmpPB = null;
	private ExecutionContext _tmpEC = null;
	
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return null;
		for( int i=0; i<roots.size(); i++ ) {
			Hop h = roots.get(i);
			roots.set(i, rule_ConstantFolding(h));
		}
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return null;
		return rule_ConstantFolding(root);
	}

	private Hop rule_ConstantFolding( Hop hop ) {
		return rConstantFoldingExpression(hop);
	}

	private Hop rConstantFoldingExpression( Hop root ) {
		if( root.isVisited() )
			return root;
		
		//recursively process childs (before replacement to allow bottom-recursion)
		//no iterator in order to prevent concurrent modification
		for( int i=0; i<root.getInput().size(); i++ ) {
			Hop h = root.getInput().get(i);
			rConstantFoldingExpression(h);
		}
		
		LiteralOp literal = null;
		
		//fold binary op if both are literals / unary op if literal
		if( root.getDataType() == DataType.SCALAR //scalar output
			&& ( isApplicableBinaryOp(root) || isApplicableUnaryOp(root) ) )
		{ 
			literal = evalScalarOperation(root); 
		}
		//fold conjunctive predicate if at least one input is literal 'false'
		else if( isApplicableFalseConjunctivePredicate(root) ) {
			literal = new LiteralOp(false);
		}
		//fold disjunctive predicate if at least one input is literal 'true'
		else if( isApplicableTrueDisjunctivePredicate(root) ) {
			literal = new LiteralOp(true);
		}
		
		//replace binary operator with folded constant
		if( literal != null ) {
			//bottom-up replacement to keep common subexpression elimination
			if( !root.getParent().isEmpty() ) { //broot is NOT a DAG root
				List<Hop> parents = new ArrayList<>(root.getParent());
				for( Hop parent : parents )
					HopRewriteUtils.replaceChildReference(parent, root, literal);
			}
			else { //broot IS a DAG root
				root = literal;
			}
		}
		
		//mark processed
		root.setVisited();
		return root;
	}
	
	/**
	 * In order to (1) prevent unexpected side effects from constant folding and
	 * (2) for simplicity with regard to arbitrary value type combinations,
	 * we use the same compilation and runtime for constant folding as we would 
	 * use for actual instruction execution. 
	 * 
	 * @param bop high-level operator
	 * @return literal op
	 */
	private LiteralOp evalScalarOperation( Hop bop ) 
	{
		//Timing time = new Timing( true );
		
		DataOp tmpWrite = new DataOp(TMP_VARNAME, bop.getDataType(),
			bop.getValueType(), bop, OpOpData.TRANSIENTWRITE, TMP_VARNAME);
		
		//generate runtime instruction
		Dag<Lop> dag = new Dag<>();
		Recompiler.rClearLops(tmpWrite); //prevent lops reuse
		Lop lops = tmpWrite.constructLops(); //reconstruct lops
		lops.addToDag( dag );
		ArrayList<Instruction> inst = dag.getJobs(null, ConfigurationManager.getDMLConfig());
		
		//execute instructions
		ExecutionContext ec = getExecutionContext();
		BasicProgramBlock pb = getProgramBlock();
		pb.setInstructions( inst );
		
		pb.execute( ec );
		
		//get scalar result (check before invocation) and create literal according
		//to observed scalar output type (not hop type) for runtime consistency
		ScalarObject so = (ScalarObject) ec.getVariable(TMP_VARNAME);
		LiteralOp literal = ScalarObjectFactory.createLiteralOp(so);
		
		//cleanup
		tmpWrite.getInput().clear();
		bop.getParent().remove(tmpWrite);
		pb.setInstructions(null);
		ec.getVariables().removeAll();
		
		//set literal properties (scalar)
		HopRewriteUtils.setOutputParametersForScalar(literal);
 		
		//System.out.println("Constant folded in "+time.stop()+"ms.");
		
		return literal;
	}
	
	private BasicProgramBlock getProgramBlock() {
		if( _tmpPB == null )
			_tmpPB = new BasicProgramBlock(new Program());
		return _tmpPB;
	}
	
	private ExecutionContext getExecutionContext() {
		if( _tmpEC == null )
			_tmpEC = ExecutionContextFactory.createContext();
		return _tmpEC;
	}
	
	private static boolean isApplicableBinaryOp( Hop hop )
	{
		ArrayList<Hop> in = hop.getInput();
		return (   hop instanceof BinaryOp 
				&& in.get(0) instanceof LiteralOp 
				&& in.get(1) instanceof LiteralOp
				&& ((BinaryOp)hop).getOp()!=OpOp2.CBIND
				&& ((BinaryOp)hop).getOp()!=OpOp2.RBIND);
		
		//string append is rejected although possible because it
		//messes up the explain runtime output due to introduced \n 
	}
	
	private static boolean isApplicableUnaryOp( Hop hop ) {
		ArrayList<Hop> in = hop.getInput();
		return (   hop instanceof UnaryOp 
				&& in.get(0) instanceof LiteralOp 
				&& ((UnaryOp)hop).getOp() != OpOp1.EXISTS
				&& ((UnaryOp)hop).getOp() != OpOp1.PRINT
				&& ((UnaryOp)hop).getOp() != OpOp1.ASSERT
				&& ((UnaryOp)hop).getOp() != OpOp1.STOP
				&& hop.getDataType() == DataType.SCALAR);
	}
	
	private static boolean isApplicableFalseConjunctivePredicate( Hop hop ) {
		ArrayList<Hop> in = hop.getInput();
		return (   HopRewriteUtils.isBinary(hop, OpOp2.AND) && hop.getDataType().isScalar()
				&& ( (in.get(0) instanceof LiteralOp && !((LiteralOp)in.get(0)).getBooleanValue())
				   ||(in.get(1) instanceof LiteralOp && !((LiteralOp)in.get(1)).getBooleanValue())) );
	}
	
	private static boolean isApplicableTrueDisjunctivePredicate( Hop hop ) {
		ArrayList<Hop> in = hop.getInput();
		return (   HopRewriteUtils.isBinary(hop, OpOp2.OR) && hop.getDataType().isScalar()
				&& ( (in.get(0) instanceof LiteralOp && ((LiteralOp)in.get(0)).getBooleanValue())
				   ||(in.get(1) instanceof LiteralOp && ((LiteralOp)in.get(1)).getBooleanValue())) );
	}
}
