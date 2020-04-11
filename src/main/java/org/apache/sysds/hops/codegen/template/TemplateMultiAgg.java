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

package org.apache.sysds.hops.codegen.template;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.hops.codegen.cplan.CNodeMultiAgg;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

public class TemplateMultiAgg extends TemplateCell 
{	
	public TemplateMultiAgg() {
		super(TemplateType.MAGG, CloseType.OPEN_VALID);
	}
	
	public TemplateMultiAgg(CloseType ctype) {
		super(TemplateType.MAGG, ctype);
	}

	@Override
	public boolean open(Hop hop) {
		//multiagg is a composite templates, which is not 
		//created via open-fuse-merge-close
		return false;
	}

	@Override
	public boolean fuse(Hop hop, Hop input) {
		return false;
	}

	@Override
	public boolean merge(Hop hop, Hop input) {
		return false;
	}

	@Override
	public CloseType close(Hop hop) {
		return CloseType.CLOSED_INVALID;
	}

	@Override
	public Pair<Hop[], CNodeTpl> constructCplan(Hop hop, CPlanMemoTable memo, boolean compileLiterals) 
	{
		//get all root nodes for multi aggregation
		MemoTableEntry multiAgg = memo.getBest(hop.getHopID(), TemplateType.MAGG);
		ArrayList<Hop> roots = new ArrayList<>();
		for( int i=0; i<3; i++ )
			if( multiAgg.isPlanRef(i) )
				roots.add(memo._hopRefs.get(multiAgg.input(i)));
		Hop.resetVisitStatus(roots);
		
		//recursively process required cplan outputs
		HashSet<Hop> inHops = new HashSet<>();
		HashMap<Long, CNode> tmp = new HashMap<>();
		for( Hop root : roots ) //use celltpl cplan construction
			super.rConstructCplan(root, memo, tmp, inHops, compileLiterals);
		Hop.resetVisitStatus(roots);
		
		//reorder inputs (ensure matrices/vectors come first) and prune literals
		//note: we order by number of cells and subsequently sparsity to ensure
		//that sparse inputs are used as the main input w/o unnecessary conversion
		Hop shared = getSparseSafeSharedInput(roots, inHops);
		Hop[] sinHops = inHops.stream()
			.filter(h -> !(h.getDataType().isScalar() && tmp.get(h.getHopID()).isLiteral()))
			.sorted(new HopInputComparator(shared)).toArray(Hop[]::new);
		
		//construct template node
		ArrayList<CNode> inputs = new ArrayList<>();
		for( Hop in : sinHops )
			inputs.add(tmp.get(in.getHopID()));
		ArrayList<CNode> outputs = new ArrayList<>();
		ArrayList<AggOp> aggOps = new ArrayList<>();
		for( Hop root : roots ) {
			CNode node = tmp.get(root.getHopID());
			if( node instanceof CNodeData //add indexing ops for sideways data inputs
				&& ((CNodeData)inputs.get(0)).getHopID() != ((CNodeData)node).getHopID() )
				node = new CNodeUnary(node, (roots.get(0).getDim2()==1) ? 
						UnaryType.LOOKUP_R : UnaryType.LOOKUP_RC);
			outputs.add(node);
			aggOps.add(TemplateUtils.getAggOp(root));
		}
		CNodeMultiAgg tpl = new CNodeMultiAgg(inputs, outputs);
		tpl.setAggOps(aggOps);
		tpl.setSparseSafe(isSparseSafe(roots, sinHops[0], 
			tpl.getOutputs(), tpl.getAggOps(), true));
		tpl.setRootNodes(roots);
		tpl.setBeginLine(hop.getBeginLine());
		
		// return cplan instance
		return new Pair<>(sinHops, tpl);
	}
	
	private Hop getSparseSafeSharedInput(ArrayList<Hop> roots, HashSet<Hop> inHops) {
		Set<Hop> tmp = inHops.stream()
			.filter(h -> h.getDataType().isMatrix())
			.collect(Collectors.toSet());
		for( Hop root : roots ) {
			root.resetVisitStatus();
			HashSet<Hop> inputs = new HashSet<>();
			rCollectSparseSafeInputs(root, inHops, inputs);
			tmp.removeIf(h -> !inputs.contains(h));
		}
		Hop.resetVisitStatus(roots);
		return tmp.isEmpty() ? null : 
			tmp.toArray(new Hop[0])[0];
	}
	
	private void rCollectSparseSafeInputs(Hop current, HashSet<Hop> inHops, HashSet<Hop> sparseInputs) {
		if( current.isVisited() || !(HopRewriteUtils.isBinary(current, OpOp2.MULT)
			|| HopRewriteUtils.isAggUnaryOp(current, AggOp.SUM, AggOp.SUM_SQ))) {
			return;
		}
		
		for( Hop c : current.getInput() ) {
			if( !inHops.contains(c) )
				rCollectSparseSafeInputs(c, inHops, sparseInputs);
			else if( c.dimsKnown(true) && MatrixBlock.evalSparseFormatInMemory(
				c.getDim1(), c.getDim2(), c.getNnz()) )
				sparseInputs.add(c);
		}
		
		current.setVisited();
	}
}
