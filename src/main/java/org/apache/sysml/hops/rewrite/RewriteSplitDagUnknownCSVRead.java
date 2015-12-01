/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;

import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.Hop.FileFormatTypes;
import org.apache.sysml.hops.Hop.VisitStatus;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.VariableSet;

/**
 * Rule: Split Hop DAG after CSV reads with unknown size. This is
 * important to create recompile hooks if format is read from mtd
 * (we are not able to split it on statementblock creation) and 
 * mtd has unknown size (which can only happen for CSV). 
 * 
 */
public class RewriteSplitDagUnknownCSVRead extends StatementBlockRewriteRule
{

	@Override
	public ArrayList<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
		throws HopsException 
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		
		//collect all unknown csv reads hops
		ArrayList<Hop> cand = new ArrayList<Hop>();
		collectCSVReadHopsUnknownSize( sb.get_hops(), cand );
		
		//split hop dag on demand
		if( !cand.isEmpty() )
		{
			try
			{
				//duplicate sb incl live variable sets
				StatementBlock sb1 = new StatementBlock();
				sb1.setDMLProg(sb.getDMLProg());
				sb1.setAllPositions(sb.getFilename(), sb.getBeginLine(), sb.getBeginColumn(), sb.getEndLine(), sb.getEndColumn());
				sb1.setLiveIn(new VariableSet());
				sb1.setLiveOut(new VariableSet());
				
				//move csv reads incl reblock to new statement block
				//(and replace original persistent read with transient read)
				ArrayList<Hop> sb1hops = new ArrayList<Hop>();			
				for( Hop c : cand )
				{
					Hop reblock = c;
					long rlen = reblock.getDim1();
					long clen = reblock.getDim2();
					long nnz = reblock.getNnz();
					long brlen = reblock.getRowsInBlock();
					long bclen = reblock.getColsInBlock();
	
					//create new transient read
					DataOp tread = new DataOp(reblock.getName(), reblock.getDataType(), reblock.getValueType(),
		                    DataOpTypes.TRANSIENTREAD, null, rlen, clen, nnz, brlen, bclen);
					HopRewriteUtils.copyLineNumbers(reblock, tread);
					
					//replace reblock with transient read
					ArrayList<Hop> parents = new ArrayList<Hop>(reblock.getParent());
					for( int i=0; i<parents.size(); i++ )
					{
						Hop parent = parents.get(i);
						int pos = HopRewriteUtils.getChildReferencePos(parent, reblock);
						HopRewriteUtils.removeChildReferenceByPos(parent, reblock, pos);
						HopRewriteUtils.addChildReference(parent, tread, pos);
					}
					
					//add reblock sub dag to first statement block
					DataOp twrite = new DataOp(reblock.getName(), reblock.getDataType(), reblock.getValueType(),
							                   reblock, DataOpTypes.TRANSIENTWRITE, null);
					twrite.setOutputParams(rlen, clen, nnz, brlen, bclen);
					HopRewriteUtils.copyLineNumbers(reblock, twrite);
					sb1hops.add(twrite);
					
					//update live in and out of new statement block (for piggybacking)
					DataIdentifier diVar = sb.variablesRead().getVariable(reblock.getName()); 
					if( diVar != null ){ //var read should always exist because persistent read
						sb1.liveOut().addVariable(reblock.getName(), new DataIdentifier(diVar));
						sb.liveIn().addVariable(reblock.getName(), new DataIdentifier(diVar));
					}
				}
				
				sb1.set_hops(sb1hops);
				sb1.updateRecompilationFlag();
				ret.add(sb1); //statement block with csv reblocks
				ret.add(sb); //statement block with remaining hops
			}
			catch(Exception ex)
			{
				throw new HopsException("Failed to split hops dag for csv read with unknown size.", ex);
			}
			LOG.debug("Applied splitDagUnknownCSVRead.");
		}
		//keep original hop dag
		else
		{
			ret.add(sb);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param roots
	 * @param cand
	 */
	private void collectCSVReadHopsUnknownSize( ArrayList<Hop> roots, ArrayList<Hop> cand )
	{
		if( roots == null )
			return;
		
		Hop.resetVisitStatus(roots);
		for( Hop root : roots )
			collectCSVReadHopsUnknownSize(root, cand);
	}
	
	/**
	 * 
	 * @param root
	 * @param cand
	 */
	private void collectCSVReadHopsUnknownSize( Hop hop, ArrayList<Hop> cand )
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;
		
		//collect persistent reads (of type csv, with unknown size)
		if( hop instanceof DataOp )
		{
			DataOp dop = (DataOp) hop;
			if(    dop.getDataOpType() == DataOpTypes.PERSISTENTREAD
				&& dop.getInputFormatType() == FileFormatTypes.CSV
				&& !dop.dimsKnown()
				&& !HopRewriteUtils.hasOnlyWriteParents(dop, true, false)
				&& !HopRewriteUtils.hasTransformParents(hop) )
			{
				cand.add(dop);
			}
		}
		
		//process children
		if( hop.getInput()!=null )
			for( Hop c : hop.getInput() )
				collectCSVReadHopsUnknownSize(c, cand);
		
		hop.setVisited(VisitStatus.DONE);
	}
}
