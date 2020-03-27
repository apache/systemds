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

import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.hops.codegen.cplan.CNodeMultiAgg;
import org.apache.sysds.hops.codegen.cplan.CNodeOuterProduct;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;

/**
 * This cplan rewriter is meant to be the central place for any cplan 
 * enhancements before code generation. These rewrites do not aim to
 * handle reorderings or other algebraic simplifications but rather
 * focus on low-level simplifications to produce better code, while
 * keeping the cplan construction of the individual templates clean
 * and without unnecessary redundancy.
 * 
 * Assumption: This rewriter should be called before CSE as these
 * rewrites potentially destroy common subexpressions.
 */
public class CPlanOpRewriter 
{
	public CNodeTpl simplifyCPlan(CNodeTpl tpl) {
		//apply template specific rewrites
		tpl = rewriteRemoveOuterNeq0(tpl); // Outer(a!=0) -> Outer(1)
		
		//apply operation specific rewrites
		if( tpl instanceof CNodeMultiAgg ) {
			ArrayList<CNode> outputs = ((CNodeMultiAgg)tpl).getOutputs();
			for( int i=0; i< outputs.size(); i++ )
				outputs.set(i, rSimplifyCNode(outputs.get(i)));
		}
		else {
			tpl.setOutput(rSimplifyCNode(tpl.getOutput()));
		}
		
		return tpl;
	}
	
	private static CNode rSimplifyCNode(CNode node) {
		//process children recursively
		for(int i=0; i<node.getInput().size(); i++)
			node.getInput().set(i, rSimplifyCNode(node.getInput().get(i)));
		
		//apply all node-local simplification rewrites
		node = rewriteRowCountNnz(node);     //rowSums(X!=0) -> rowNnz(X)
		node = rewriteRowSumSq(node);        //rowSums(X^2) -> rowSumSqs(X)
		node = rewriteBinaryPow2(node);      //x^2 -> x*x
		node = rewriteBinaryPow2Vect(node);  //X^2 -> X*X
		node = rewriteBinaryMult2(node);     //x*2 -> x+x;
		node = rewriteBinaryMult2Vect(node); //X*2 -> X+X;
		
		return node;
	}
	
	private static CNode rewriteRowCountNnz(CNode node) {
		return (TemplateUtils.isUnary(node, UnaryType.ROW_SUMS)
			&& TemplateUtils.isBinary(node.getInput().get(0), BinType.VECT_NOTEQUAL_SCALAR)
			&& node.getInput().get(0).getInput().get(1).isLiteral()
			&& node.getInput().get(0).getInput().get(1).getVarname().equals("0")) ?
			new CNodeUnary(node.getInput().get(0).getInput().get(0), UnaryType.ROW_COUNTNNZS) : node;
	}
	
	private static CNode rewriteRowSumSq(CNode node) {
		return (TemplateUtils.isUnary(node, UnaryType.ROW_SUMS)
			&& TemplateUtils.isBinary(node.getInput().get(0), BinType.VECT_POW_SCALAR)
			&& node.getInput().get(0).getInput().get(1).isLiteral()
			&& node.getInput().get(0).getInput().get(1).getVarname().equals("2")) ?
			new CNodeUnary(node.getInput().get(0).getInput().get(0), UnaryType.ROW_SUMSQS) : node;
	}

	private static CNode rewriteBinaryPow2(CNode node) {
		return (TemplateUtils.isBinary(node, BinType.POW) 
			&& node.getInput().get(1).isLiteral()
			&& node.getInput().get(1).getVarname().equals("2")) ?
			new CNodeUnary(node.getInput().get(0), UnaryType.POW2) : node;
	}
	
	private static CNode rewriteBinaryPow2Vect(CNode node) {
		return (TemplateUtils.isBinary(node, BinType.VECT_POW_SCALAR) 
			&& node.getInput().get(1).isLiteral()
			&& node.getInput().get(1).getVarname().equals("2")) ?
			new CNodeUnary(node.getInput().get(0), UnaryType.VECT_POW2) : node;
	}
	
	private static CNode rewriteBinaryMult2(CNode node) {
		return (TemplateUtils.isBinary(node, BinType.MULT) 
			&& node.getInput().get(1).isLiteral()
			&& node.getInput().get(1).getVarname().equals("2")) ?
			new CNodeUnary(node.getInput().get(0), UnaryType.MULT2) : node;
	}
	
	private static CNode rewriteBinaryMult2Vect(CNode node) {
		return (TemplateUtils.isBinary(node, BinType.VECT_MULT) 
			&& node.getInput().get(1).isLiteral()
			&& node.getInput().get(1).getVarname().equals("2")) ?
			new CNodeUnary(node.getInput().get(0), UnaryType.VECT_MULT2) : node;
	}
	
	private static CNodeTpl rewriteRemoveOuterNeq0(CNodeTpl tpl) {
		if( tpl instanceof CNodeOuterProduct )
			rFindAndRemoveBinaryMS(tpl.getOutput(), (CNodeData)
				tpl.getInput().get(0), BinType.NOTEQUAL, "0", "1");
		return tpl;
	}
	
	private static void rFindAndRemoveBinaryMS(CNode node, CNodeData mainInput, BinType type, String lit, String replace) {
		for( int i=0; i<node.getInput().size(); i++ ) {
			CNode tmp = node.getInput().get(i);
			if( TemplateUtils.isBinary(tmp, type) && tmp.getInput().get(1).isLiteral()
				&& tmp.getInput().get(1).getVarname().equals(lit)
				&& tmp.getInput().get(0) instanceof CNodeData
				&& ((CNodeData)tmp.getInput().get(0)).getHopID()==mainInput.getHopID() )
			{
				CNodeData cnode = new CNodeData(new LiteralOp(replace));
				cnode.setLiteral(true);
				node.getInput().set(i, cnode);
			}
			else
				rFindAndRemoveBinaryMS(tmp, mainInput, type, lit, replace);
		}
	}
}
