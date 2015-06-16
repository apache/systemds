/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.VariableSet;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;

/**
 * Rule: Insert checkpointing operations for caching purposes. Currently, we
 * follow a heuristic of checkpointing (1) all variables used read-only in loops,
 * and (2) intermediates used by multiple consumers.
 * 
 * TODO (2) implement injection for multiple consumers (local and global).
 * 
 */
public class RewriteInjectSparkLoopCheckpointing extends StatementBlockRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	@Override
	public ArrayList<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
		throws HopsException 
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		
		if( !OptimizerUtils.isSparkExecutionMode() ) 
		{
			ret.add(sb); // nothing to do here
			return ret; //return original statement block
		}
		
		//1) We currently add checkpoint operations without information about the global program structure,
		//this assumes that redundant checkpointing is prevented at runtime level (instruction-level)
		//2) Also, we do not take size information into account right now. This means that all candidates
		//are checkpointed even if they are only used by CP operations.
		
		int blocksize = state.getBlocksize(); //blocksize set by reblock rewrite
		
		//apply rewrite for while and for (the decision for parfor loops is deferred until parfor
		//optimization because otherwise we would prevent remote parfor)
		//TODO this needs a more detailed treatment, which will be introduced with the generalization of reblockop/dataop 
		if( (   sb instanceof WhileStatementBlock 
			 || sb instanceof ForStatementBlock && !(sb instanceof ParForStatementBlock)) ) 
		{
			//step 1: determine checkpointing candidates
			ArrayList<String> candidates = new ArrayList<String>(); 
			VariableSet read = sb.variablesRead();
			VariableSet updated = sb.variablesUpdated();
			
			for( String rvar : read.getVariableNames() )
				if( !updated.containsVariable(rvar) && read.getVariable(rvar).getDataType()==DataType.MATRIX )
					candidates.add(rvar);
			
			//step 2: insert statementblock with checkpointing operations
			if( !candidates.isEmpty() ) //existing candidates
			{
				StatementBlock sb0 = new StatementBlock();
				sb0.setDMLProg(sb.getDMLProg());
				ArrayList<Hop> hops = new ArrayList<Hop>();
				VariableSet livein = new VariableSet();
				VariableSet liveout = new VariableSet();
				for( String var : candidates ) 
				{
					DataIdentifier dat = read.getVariable(var);
					DataOp tread = new DataOp(var, DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.TRANSIENTREAD, 
							            dat.getFilename(), dat.getDim1(), dat.getDim2(), dat.getNnz(), blocksize, blocksize);
					tread.setRequiresCheckpoint( true );
					DataOp twrite = new DataOp(var, DataType.MATRIX, ValueType.DOUBLE, tread, DataOpTypes.TRANSIENTWRITE, null);
					HopRewriteUtils.setOutputParameters(twrite, dat.getDim1(), dat.getDim2(), blocksize, blocksize, dat.getNnz());					
					hops.add(twrite);
					livein.addVariable(var, read.getVariable(var));
					liveout.addVariable(var, read.getVariable(var));
				}
				sb0.set_hops(hops);
				sb0.setLiveIn(livein);
				sb0.setLiveOut(liveout);
				ret.add(sb0);
			}
		}
			
		//add original statement block to end
		ret.add(sb);
		
		return ret;
	}
}
