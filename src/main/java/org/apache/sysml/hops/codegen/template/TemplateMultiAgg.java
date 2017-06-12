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

package org.apache.sysml.hops.codegen.template;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysml.hops.codegen.cplan.CNodeMultiAgg;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.runtime.matrix.data.Pair;

public class TemplateMultiAgg extends TemplateCell 
{	
	public TemplateMultiAgg() {
		super(TemplateType.MultiAggTpl, false);
	}
	
	public TemplateMultiAgg(boolean closed) {
		super(TemplateType.MultiAggTpl, closed);
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

	public Pair<Hop[], CNodeTpl> constructCplan(Hop hop, CPlanMemoTable memo, boolean compileLiterals) 
	{
		//get all root nodes for multi aggregation
		MemoTableEntry multiAgg = memo.getBest(hop.getHopID(), TemplateType.MultiAggTpl);
		ArrayList<Hop> roots = new ArrayList<Hop>();
		for( int i=0; i<3; i++ )
			if( multiAgg.isPlanRef(i) )
				roots.add(memo._hopRefs.get(multiAgg.input(i)));
		Hop.resetVisitStatus(roots);
		
		//recursively process required cplan outputs
		HashSet<Hop> inHops = new HashSet<Hop>();
		HashMap<Long, CNode> tmp = new HashMap<Long, CNode>();
		for( Hop root : roots ) //use celltpl cplan construction
			super.rConstructCplan(root, memo, tmp, inHops, compileLiterals);
		Hop.resetVisitStatus(roots);
		
		//reorder inputs (ensure matrices/vectors come first) and prune literals
		//note: we order by number of cells and subsequently sparsity to ensure
		//that sparse inputs are used as the main input w/o unnecessary conversion
		List<Hop> sinHops = inHops.stream()
			.filter(h -> !(h.getDataType().isScalar() && tmp.get(h.getHopID()).isLiteral()))
			.sorted(new HopInputComparator()).collect(Collectors.toList());
		
		//construct template node
		ArrayList<CNode> inputs = new ArrayList<CNode>();
		for( Hop in : sinHops )
			inputs.add(tmp.get(in.getHopID()));
		ArrayList<CNode> outputs = new ArrayList<CNode>();
		ArrayList<AggOp> aggOps = new ArrayList<AggOp>();
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
		tpl.setRootNodes(roots);
		tpl.setBeginLine(hop.getBeginLine());
		
		// return cplan instance
		return new Pair<Hop[],CNodeTpl>(sinHops.toArray(new Hop[0]), tpl);
	}
}
