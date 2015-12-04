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
import java.util.HashMap;
import java.util.Set;

import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.VisitStatus;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;

/**
 * 
 * 
 */
public class OptTreePlanChecker 
{
	
	/**
	 * 
	 * @param pb
	 * @param sb
	 * @param fnStack
	 * @throws HopsException
	 * @throws DMLRuntimeException
	 */
	public static void checkProgramCorrectness( ProgramBlock pb, StatementBlock sb, Set<String> fnStack ) 
		throws HopsException, DMLRuntimeException
	{
		Program prog = pb.getProgram();
		DMLProgram dprog = sb.getDMLProg();
		
		if (pb instanceof FunctionProgramBlock && sb instanceof FunctionStatementBlock )
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for( int i=0; i<fpb.getChildBlocks().size(); i++ )
			{
				ProgramBlock pbc = fpb.getChildBlocks().get(i);
				StatementBlock sbc = fstmt.getBody().get(i);
				checkProgramCorrectness(pbc, sbc, fnStack);
			}
			//checkLinksProgramStatementBlock(fpb, fsb);
		}
		else if (pb instanceof WhileProgramBlock && sb instanceof WhileStatementBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
			checkHopDagCorrectness(prog, dprog, wsb.getPredicateHops(), wpb.getPredicate(), fnStack);
			for( int i=0; i<wpb.getChildBlocks().size(); i++ )
			{
				ProgramBlock pbc = wpb.getChildBlocks().get(i);
				StatementBlock sbc = wstmt.getBody().get(i);
				checkProgramCorrectness(pbc, sbc, fnStack);
			}
			checkLinksProgramStatementBlock(wpb, wsb);
		}	
		else if (pb instanceof IfProgramBlock && sb instanceof IfStatementBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock) pb;
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement) isb.getStatement(0);
			checkHopDagCorrectness(prog, dprog, isb.getPredicateHops(), ipb.getPredicate(), fnStack);
			for( int i=0; i<ipb.getChildBlocksIfBody().size(); i++ ) {
				ProgramBlock pbc = ipb.getChildBlocksIfBody().get(i);
				StatementBlock sbc = istmt.getIfBody().get(i);
				checkProgramCorrectness(pbc, sbc, fnStack);
			}
			for( int i=0; i<ipb.getChildBlocksElseBody().size(); i++ ) {			
				ProgramBlock pbc = ipb.getChildBlocksElseBody().get(i);
				StatementBlock sbc = istmt.getElseBody().get(i);
				checkProgramCorrectness(pbc, sbc, fnStack);
			}
			checkLinksProgramStatementBlock(ipb, isb);
		}
		else if (pb instanceof ForProgramBlock && sb instanceof ForStatementBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock) pb;
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement) sb.getStatement(0);
			checkHopDagCorrectness(prog, dprog, fsb.getFromHops(), fpb.getFromInstructions(), fnStack);
			checkHopDagCorrectness(prog, dprog, fsb.getToHops(), fpb.getToInstructions(), fnStack);
			checkHopDagCorrectness(prog, dprog, fsb.getIncrementHops(), fpb.getIncrementInstructions(), fnStack);
			for( int i=0; i<fpb.getChildBlocks().size(); i++ ) {			
				ProgramBlock pbc = fpb.getChildBlocks().get(i);
				StatementBlock sbc = fstmt.getBody().get(i);
				checkProgramCorrectness(pbc, sbc, fnStack);
			}
			checkLinksProgramStatementBlock(fpb, fsb);
		}
		else
		{
			checkHopDagCorrectness(prog, dprog, sb.get_hops(), pb.getInstructions(), fnStack);
			//checkLinksProgramStatementBlock(pb, sb);
		}
		
	}
	
	/**
	 * 
	 * @param prog
	 * @param dprog
	 * @param roots
	 * @param inst
	 * @param fnStack
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 */
	private static void checkHopDagCorrectness( Program prog, DMLProgram dprog, ArrayList<Hop> roots, ArrayList<Instruction> inst, Set<String> fnStack ) 
		throws DMLRuntimeException, HopsException
	{
		if( roots != null )
			for( Hop hop : roots )
				checkHopDagCorrectness(prog, dprog, hop, inst, fnStack);
	}
	
	/**
	 * 
	 * @param prog
	 * @param dprog
	 * @param root
	 * @param inst
	 * @param fnStack
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 */
	private static void checkHopDagCorrectness( Program prog, DMLProgram dprog, Hop root, ArrayList<Instruction> inst, Set<String> fnStack ) 
		throws DMLRuntimeException, HopsException
	{
		//set of checks to perform
		checkFunctionNames(prog, dprog, root, inst, fnStack);
	}
	
	/**
	 * 
	 * @param pb
	 * @param sb
	 * @throws DMLRuntimeException
	 */
	private static void checkLinksProgramStatementBlock( ProgramBlock pb, StatementBlock sb ) 
		throws DMLRuntimeException
	{
		if( pb.getStatementBlock() != sb )
			throw new DMLRuntimeException("Links between programblocks and statementblocks are incorrect ("+pb+").");
	}
	
	/**
	 * 
	 * @param prog
	 * @param dprog
	 * @param root
	 * @param inst
	 * @param fnStack
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 */
	private static void checkFunctionNames( Program prog, DMLProgram dprog, Hop root, ArrayList<Instruction> inst, Set<String> fnStack ) 
		throws DMLRuntimeException, HopsException
	{
		//reset visit status of dag
		root.resetVisitStatus();
		
		//get all function op in this dag
		HashMap<String, FunctionOp> fops = new HashMap<String, FunctionOp>();
		getAllFunctionOps(root, fops);
		
		for( Instruction linst : inst )
			if( linst instanceof FunctionCallCPInstruction )
			{
				FunctionCallCPInstruction flinst = (FunctionCallCPInstruction) linst;
				String fnamespace = flinst.getNamespace();
				String fname = flinst.getFunctionName();
				String key = DMLProgram.constructFunctionKey(fnamespace, fname);
				
				//check 1: instruction name equal to hop name
				if( !fops.containsKey(key) )
					throw new DMLRuntimeException( "Function Check: instruction and hop names differ ("+key+", "+fops.keySet()+")" );
				
				//check 2: function exists
				if( !prog.getFunctionProgramBlocks().containsKey(key) )
					throw new DMLRuntimeException( "Function Check: function does not exits ("+key+")" );
	
				//check 3: recursive program check
				FunctionProgramBlock fpb = prog.getFunctionProgramBlock(fnamespace, fname);
				FunctionStatementBlock fsb = dprog.getFunctionStatementBlock(fnamespace, fname);
				if( !fnStack.contains(key) )
				{
					fnStack.add(key);
					checkProgramCorrectness(fpb, fsb, fnStack);
					fnStack.remove(key);
				}
			}
	}
	
	/**
	 * 
	 * @param hop
	 * @param memo
	 */
	private static void getAllFunctionOps( Hop hop, HashMap<String, FunctionOp> memo )
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;
		
		//process functionop
		if( hop instanceof FunctionOp )
		{
			FunctionOp fop = (FunctionOp) hop;
			String key = DMLProgram.constructFunctionKey(fop.getFunctionNamespace(),fop.getFunctionName());
			memo.put(key, fop);
		}
		
		//process children
		for( Hop in : hop.getInput() )
			getAllFunctionOps(in, memo);
		
		hop.setVisited(VisitStatus.DONE);
	}
}
