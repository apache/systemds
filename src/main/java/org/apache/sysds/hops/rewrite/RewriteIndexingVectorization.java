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

import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LeftIndexingOp;
import org.apache.sysds.hops.LiteralOp;

/**
 * Rule: Indexing vectorization. This rewrite rule set simplifies
 * multiple right / left indexing accesses within a DAG into row/column
 * index accesses, which is beneficial for two reasons: (1) it is an 
 * enabler for later row/column partitioning, and (2) it reduces the number
 * of operations over potentially large data (i.e., prevents unnecessary MR 
 * operations and reduces pressure on the buffer pool due to copy on write
 * on left indexing).
 * 
 */
public class RewriteIndexingVectorization extends HopRewriteRule
{
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return roots;
		for( Hop h : roots )
			rule_IndexingVectorization( h );
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return root;
		rule_IndexingVectorization( root );
		return root;
	}

	private void rule_IndexingVectorization( Hop hop ) {
		if(hop.isVisited())
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++)
		{
			Hop hi = hop.getInput().get(i);
			
			//apply indexing vectorization rewrites
			//MB: disabled right indexing rewrite because (1) piggybacked in MR anyway, (2) usually
			//not too much overhead, and (3) makes literal replacement more difficult
			hi = vectorizeRightLeftIndexingChains(hi); //e.g., B[,1]=A[,1]; B[,2]=A[2]; -> B[,1:2] = A[,1:2]
			//vectorizeRightIndexing( hi ); //e.g., multiple rightindexing X[i,1], X[i,3] -> X[i,];
			hi = vectorizeLeftIndexing( hi );  //e.g., multiple left indexing X[i,1], X[i,3] -> X[i,]; 
			
			//process childs recursively after rewrites
			rule_IndexingVectorization( hi );
		}

		hop.setVisited();
	}

	private static Hop vectorizeRightLeftIndexingChains(Hop hi) {
		//check for valid root operator
		if( !(hi instanceof LeftIndexingOp 
			&& hi.getInput().get(1) instanceof IndexingOp
			&& hi.getInput().get(1).getParent().size()==1) )
			return hi;
		LeftIndexingOp lix0 = (LeftIndexingOp) hi;
		IndexingOp rix0 = (IndexingOp) hi.getInput().get(1);
		if( !(lix0.isRowLowerEqualsUpper() || lix0.isColLowerEqualsUpper())
			|| lix0.isRowLowerEqualsUpper() != rix0.isRowLowerEqualsUpper()
			|| lix0.isColLowerEqualsUpper() != rix0.isColLowerEqualsUpper())
			return hi;
		boolean row = lix0.isRowLowerEqualsUpper();
		if( !( (row ? HopRewriteUtils.isFullRowIndexing(lix0) : HopRewriteUtils.isFullColumnIndexing(lix0))
			&& (row ? HopRewriteUtils.isFullRowIndexing(rix0) : HopRewriteUtils.isFullColumnIndexing(rix0))) )
			return hi;
		
		//determine consecutive left-right indexing chains for rows/columns
		List<LeftIndexingOp> lix = new ArrayList<>(); lix.add(lix0);
		List<IndexingOp> rix = new ArrayList<>(); rix.add(rix0);
		LeftIndexingOp clix = lix0;
		IndexingOp crix = rix0;
		while( isConsecutiveLeftRightIndexing(clix, crix, clix.getInput().get(0))
			&& clix.getInput().get(0).getParent().size()==1
			&& clix.getInput().get(0).getInput().get(1).getParent().size()==1 ) {
			clix = (LeftIndexingOp)clix.getInput().get(0);
			crix = (IndexingOp)clix.getInput().get(1);
			lix.add(clix); rix.add(crix);
		}
		
		//rewrite pattern if at least two consecutive pairs
		if( lix.size() >= 2 ) {
			IndexingOp rixn = rix.get(rix.size()-1);
			Hop rlrix = rixn.getInput().get(1);
			Hop rurix = row ? HopRewriteUtils.createBinary(rlrix, new LiteralOp(rix.size()-1), OpOp2.PLUS) : rixn.getInput().get(2);
			Hop clrix = rixn.getInput().get(3);
			Hop curix = row ? rixn.getInput().get(4) : HopRewriteUtils.createBinary(clrix, new LiteralOp(rix.size()-1), OpOp2.PLUS);
			IndexingOp rixNew = HopRewriteUtils.createIndexingOp(rixn.getInput().get(0), rlrix, rurix, clrix, curix);
			
			LeftIndexingOp lixn = lix.get(rix.size()-1);
			Hop rllix = lixn.getInput().get(2);
			Hop rulix = row ? HopRewriteUtils.createBinary(rllix, new LiteralOp(lix.size()-1), OpOp2.PLUS) : lixn.getInput().get(3);
			Hop cllix = lixn.getInput().get(4);
			Hop culix = row ? lixn.getInput().get(5) : HopRewriteUtils.createBinary(cllix, new LiteralOp(lix.size()-1), OpOp2.PLUS);
			LeftIndexingOp lixNew = HopRewriteUtils.createLeftIndexingOp(lixn.getInput().get(0), rixNew, rllix, rulix, cllix, culix);
			
			//rewire parents and childs
			HopRewriteUtils.replaceChildReference(hi.getParent().get(0), hi, lixNew);
			for( int i=0; i<lix.size(); i++ ) {
				HopRewriteUtils.removeAllChildReferences(lix.get(i));
				HopRewriteUtils.removeAllChildReferences(rix.get(i));
			}
			
			hi = lixNew;
			LOG.debug("Applied vectorizeRightLeftIndexingChains (line "+hi.getBeginLine()+")");
		}
		
		return hi;
	}
	

	private static boolean isConsecutiveLeftRightIndexing(LeftIndexingOp lix, IndexingOp rix, Hop input) {
		if( !(input instanceof LeftIndexingOp 
			&& input.getInput().get(1) instanceof IndexingOp) )
			return false;
		boolean row = lix.isRowLowerEqualsUpper();
		LeftIndexingOp lix2 = (LeftIndexingOp) input;
		IndexingOp rix2 = (IndexingOp) input.getInput().get(1);
		//check row/column access with full row/column indexing
		boolean access = (row ? HopRewriteUtils.isFullRowIndexing(lix2) && HopRewriteUtils.isFullRowIndexing(rix2) :
			HopRewriteUtils.isFullColumnIndexing(lix2) && HopRewriteUtils.isFullColumnIndexing(rix2));
		//check equivalent right indexing inputs
		boolean rixInputs = (rix.getInput().get(0) == rix2.getInput().get(0));
		//check consecutive access
		boolean consecutive = (row ? HopRewriteUtils.isConsecutiveIndex(lix2.getInput().get(2), lix.getInput().get(2))
				&& HopRewriteUtils.isConsecutiveIndex(rix2.getInput().get(1), rix.getInput().get(1)) : 
			HopRewriteUtils.isConsecutiveIndex(lix2.getInput().get(4), lix.getInput().get(4))
				&& HopRewriteUtils.isConsecutiveIndex(rix2.getInput().get(3), rix.getInput().get(3)));
		return access && rixInputs && consecutive;
	}
	
	/**
	 * Note: unnecessary row or column indexing then later removed via
	 * dynamic rewrites
	 * 
	 * @param hop high-level operator
	 */
	@SuppressWarnings("unused")
	private static void vectorizeRightIndexing( Hop hop )
	{
		if( hop instanceof IndexingOp ) //right indexing
		{
			IndexingOp ihop0 = (IndexingOp) hop;
			boolean isSingleRow = ihop0.isRowLowerEqualsUpper();
			boolean isSingleCol = ihop0.isColLowerEqualsUpper();
			boolean appliedRow = false;
			
			//search for multiple indexing in same row
			if( isSingleRow && isSingleCol ){
				Hop input = ihop0.getInput().get(0);
				//find candidate set
				//dependence on common subexpression elimination to find equal input / row expression
				ArrayList<Hop> ihops = new ArrayList<Hop>();
				ihops.add(ihop0);
				for( Hop c : input.getParent() ){
					if( c != ihop0 && c instanceof IndexingOp && c.getInput().get(0) == input
						&& ((IndexingOp) c).isRowLowerEqualsUpper() && !c.isScalar()
						&& c.getInput().get(1)==ihop0.getInput().get(1) )
					{
						ihops.add( c );
					}
				}
				//apply rewrite if found candidates
				if( ihops.size() > 1 ){
					//new row indexing operator
					IndexingOp newRix = new IndexingOp("tmp", input.getDataType(), input.getValueType(), input, 
						ihop0.getInput().get(1), ihop0.getInput().get(1), new LiteralOp(1), 
						HopRewriteUtils.createValueHop(input, false), true, false); 
					HopRewriteUtils.setOutputParameters(newRix, -1, -1, input.getBlocksize(), -1);
					newRix.refreshSizeInformation();
					//rewire current operator and all candidates
					for( Hop c : ihops ) {
						HopRewriteUtils.removeChildReference(c, input); //input data
						HopRewriteUtils.addChildReference(c, newRix, 0);
						HopRewriteUtils.removeChildReferenceByPos(c, c.getInput().get(1),1); //row lower expr
						HopRewriteUtils.addChildReference(c, new LiteralOp(1), 1);
						HopRewriteUtils.removeChildReferenceByPos(c, c.getInput().get(2),2); //row upper expr
						HopRewriteUtils.addChildReference(c, new LiteralOp(1), 2);
						c.refreshSizeInformation();
					}
					
					appliedRow = true;
					LOG.debug("Applied vectorizeRightIndexingRow");
				}
			}
			
			//search for multiple indexing in same col
			if( isSingleRow && isSingleCol && !appliedRow ){
				Hop input = ihop0.getInput().get(0);
				//find candidate set
				//dependence on common subexpression elimination to find equal input / row expression
				ArrayList<Hop> ihops = new ArrayList<Hop>();
				ihops.add(ihop0);
				for( Hop c : input.getParent() ){
					if( c != ihop0 && c instanceof IndexingOp && c.getInput().get(0) == input
					   && ((IndexingOp) c).isColLowerEqualsUpper() && !c.isScalar()
					   && c.getInput().get(3)==ihop0.getInput().get(3) )
					{
						ihops.add( c );
					}
				}
				//apply rewrite if found candidates
				if( ihops.size() > 1 ){
					//new row indexing operator
					IndexingOp newRix = new IndexingOp("tmp", input.getDataType(), input.getValueType(), input, 
						new LiteralOp(1), HopRewriteUtils.createValueHop(input, true),
						ihop0.getInput().get(3), ihop0.getInput().get(3), false, true); 
					HopRewriteUtils.setOutputParameters(newRix, -1, -1, input.getBlocksize(), -1);
					newRix.refreshSizeInformation();
					//rewire current operator and all candidates
					for( Hop c : ihops ) {
						HopRewriteUtils.removeChildReference(c, input); //input data
						HopRewriteUtils.addChildReference(c, newRix, 0);
						HopRewriteUtils.replaceChildReference(c, c.getInput().get(3), new LiteralOp(1), 3); //col lower expr
						HopRewriteUtils.replaceChildReference(c, c.getInput().get(4), new LiteralOp(1), 4); //col upper expr 
						c.refreshSizeInformation();
					}

					LOG.debug("Applied vectorizeRightIndexingCol");
				}
			}
		}
	}
	
	private static Hop vectorizeLeftIndexing( Hop hop )
	{
		Hop ret = hop;
		
		if( hop instanceof LeftIndexingOp ) //left indexing
		{
			LeftIndexingOp ihop0 = (LeftIndexingOp) hop;
			boolean isSingleRow = ihop0.isRowLowerEqualsUpper();
			boolean isSingleCol = ihop0.isColLowerEqualsUpper();
			boolean appliedRow = false;
			
			if( isSingleRow && isSingleCol )
			{
				//collect simple chains (w/o multiple consumers) of left indexing ops
				ArrayList<Hop> ihops = new ArrayList<>();
				ihops.add(ihop0);
				Hop current = ihop0;
				while( current.getInput().get(0) instanceof LeftIndexingOp ) {
					LeftIndexingOp tmp = (LeftIndexingOp) current.getInput().get(0);
					if(    tmp.getParent().size()>1  //multiple consumers, i.e., not a simple chain
						|| !tmp.isRowLowerEqualsUpper() //row merge not applicable
						|| tmp.getInput().get(2) != ihop0.getInput().get(2) //not the same row
						|| tmp.getInput().get(0).getDim2() <= 1 ) //target is single column or unknown 
					{
						break;
					}
					ihops.add( tmp );
					current = tmp;
				}
				
				//apply rewrite if found candidates
				if( ihops.size() > 1 ){
					Hop input = current.getInput().get(0);
					Hop rowExpr = ihop0.getInput().get(2); //keep before reset
					
					//new row indexing operator
					IndexingOp newRix = new IndexingOp("tmp1", input.getDataType(), input.getValueType(), 
						input, rowExpr, rowExpr, new LiteralOp(1), 
						HopRewriteUtils.createValueHop(input, false), true, false); 
					HopRewriteUtils.setOutputParameters(newRix, -1, -1, input.getBlocksize(), -1);
					newRix.refreshSizeInformation();
					//reset visit status of copied hops (otherwise hidden by left indexing)
					for( Hop c : newRix.getInput() )
						c.resetVisitStatus();
					
					//rewrite bottom left indexing operator
					HopRewriteUtils.removeChildReference(current, input); //input data
					HopRewriteUtils.addChildReference(current, newRix, 0);
					
					//reset row index all candidates and refresh sizes (bottom-up)
					for( int i=ihops.size()-1; i>=0; i-- ) {
						Hop c = ihops.get(i);
						HopRewriteUtils.replaceChildReference(c, c.getInput().get(2), new LiteralOp(1), 2); //row lower expr
						HopRewriteUtils.replaceChildReference(c, c.getInput().get(3), new LiteralOp(1), 3); //row upper expr
						((LeftIndexingOp)c).setRowLowerEqualsUpper(true);
						c.refreshSizeInformation();
					}
					
					//new row left indexing operator (for all parents, only intermediates are guaranteed to have 1 parent)
					//(note: it's important to clone the parent list before creating newLix on top of ihop0)
					List<Hop> ihop0parents = new ArrayList<>(ihop0.getParent());
					List<Integer> ihop0parentsPos = new ArrayList<>();
					for( Hop parent : ihop0parents ) {
						int posp = HopRewriteUtils.getChildReferencePos(parent, ihop0);
						HopRewriteUtils.removeChildReferenceByPos(parent, ihop0, posp); //input data
						ihop0parentsPos.add(posp);
					}
					
					LeftIndexingOp newLix = new LeftIndexingOp("tmp2", input.getDataType(), input.getValueType(), 
						input, ihop0, rowExpr, rowExpr, new LiteralOp(1), 
						HopRewriteUtils.createValueHop(input, false), true, false); 
					HopRewriteUtils.setOutputParameters(newLix, -1, -1, input.getBlocksize(), -1);
					newLix.refreshSizeInformation();
					//reset visit status of copied hops (otherwise hidden by left indexing)
					for( Hop c : newLix.getInput() )
						c.resetVisitStatus();
					
					for( int i=0; i<ihop0parentsPos.size(); i++ ) {
						Hop parent = ihop0parents.get(i);
						int posp = ihop0parentsPos.get(i);
						HopRewriteUtils.addChildReference(parent, newLix, posp);
					}
					
					appliedRow = true;
					ret = newLix;
					LOG.debug("Applied vectorizeLeftIndexingRow for hop "+hop.getHopID());
				}
			}
			
			if( isSingleRow && isSingleCol && !appliedRow )
			{
				
				//collect simple chains (w/o multiple consumers) of left indexing ops
				ArrayList<Hop> ihops = new ArrayList<>();
				ihops.add(ihop0);
				Hop current = ihop0;
				while( current.getInput().get(0) instanceof LeftIndexingOp ) {
					LeftIndexingOp tmp = (LeftIndexingOp) current.getInput().get(0);
					if(    tmp.getParent().size()>1  //multiple consumers, i.e., not a simple chain
						|| !tmp.isColLowerEqualsUpper() //row merge not applicable
						|| tmp.getInput().get(4) != ihop0.getInput().get(4)  //not the same col
						|| tmp.getInput().get(0).getDim1() <= 1 )  //target is single row or unknown
					{
						break;
					}
					ihops.add( tmp );
					current = tmp;
				}
				
				//apply rewrite if found candidates
				if( ihops.size() > 1 ){
					Hop input = current.getInput().get(0);
					Hop colExpr = ihop0.getInput().get(4); //keep before reset
					
					//new row indexing operator
					IndexingOp newRix = new IndexingOp("tmp1", input.getDataType(), input.getValueType(),
						input, new LiteralOp(1), HopRewriteUtils.createValueHop(input, true),
						colExpr, colExpr, false, true); 
					HopRewriteUtils.setOutputParameters(newRix, -1, -1, input.getBlocksize(), -1);
					newRix.refreshSizeInformation();
					//reset visit status of copied hops (otherwise hidden by left indexing)
					for( Hop c : newRix.getInput() )
						c.resetVisitStatus();
					
					//rewrite bottom left indexing operator
					HopRewriteUtils.removeChildReference(current, input); //input data
					HopRewriteUtils.addChildReference(current, newRix, 0);
					
					//reset col index all candidates and refresh sizes (bottom-up)
					for( int i=ihops.size()-1; i>=0; i-- ) {
						Hop c = ihops.get(i);
						HopRewriteUtils.replaceChildReference(c, c.getInput().get(4), new LiteralOp(1), 4); //col lower expr
						HopRewriteUtils.replaceChildReference(c, c.getInput().get(5), new LiteralOp(1), 5); //col upper expr
						((LeftIndexingOp)c).setColLowerEqualsUpper(true);
						c.refreshSizeInformation();
					}
					
					//new row left indexing operator (for all parents, only intermediates are guaranteed to have 1 parent)
					//(note: it's important to clone the parent list before creating newLix on top of ihop0)
					List<Hop> ihop0parents = new ArrayList<>(ihop0.getParent());
					List<Integer> ihop0parentsPos = new ArrayList<>();
					for( Hop parent : ihop0parents ) {
						int posp = HopRewriteUtils.getChildReferencePos(parent, ihop0);
						HopRewriteUtils.removeChildReferenceByPos(parent, ihop0, posp); //input data
						ihop0parentsPos.add(posp);
					}
					
					LeftIndexingOp newLix = new LeftIndexingOp("tmp2", input.getDataType(), input.getValueType(), 
						input, ihop0, new LiteralOp(1), HopRewriteUtils.createValueHop(input, true), 
						colExpr, colExpr, false, true); 
					HopRewriteUtils.setOutputParameters(newLix, -1, -1, input.getBlocksize(), -1);
					newLix.refreshSizeInformation();
					//reset visit status of copied hops (otherwise hidden by left indexing)
					for( Hop c : newLix.getInput() )
						c.resetVisitStatus();
					
					for( int i=0; i<ihop0parentsPos.size(); i++ ) {
						Hop parent = ihop0parents.get(i);
						int posp = ihop0parentsPos.get(i);
						HopRewriteUtils.addChildReference(parent, newLix, posp);
					}
					
					ret = newLix;
					LOG.debug("Applied vectorizeLeftIndexingCol for hop "+hop.getHopID());
				}
			}
		}
		
		return ret;
	}
}
