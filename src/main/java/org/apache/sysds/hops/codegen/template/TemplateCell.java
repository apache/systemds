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
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.DnnOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeCell;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary;
import org.apache.sysds.hops.codegen.cplan.CNodeTpl;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.cplan.CNodeTernary.TernaryType;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysds.hops.codegen.template.CPlanMemoTable.MemoTableEntry;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpDnn;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.runtime.matrix.data.Pair;

public class TemplateCell extends TemplateBase 
{	
	private static final AggOp[] SUPPORTED_AGG = 
			new AggOp[]{AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX, AggOp.PROD};
	
	public TemplateCell() {
		super(TemplateType.CELL);
	}
	
	public TemplateCell(CloseType ctype) {
		super(TemplateType.CELL, ctype);
	}
	
	public TemplateCell(TemplateType type, CloseType ctype) {
		super(type, ctype);
	}
	
	@Override
	public boolean open(Hop hop) {
		return hop.dimsKnown() && isValidOperation(hop)
				&& !(hop.getDim1()==1 && hop.getDim2()==1) 
			|| (hop instanceof IndexingOp && hop.getInput().get(0).getDim2() >= 0
				&& (((IndexingOp)hop).isColLowerEqualsUpper() || hop.getDim2()==1)
				&& !((IndexingOp)hop).isScalarOutput())
			|| (HopRewriteUtils.isDataGenOpWithLiteralInputs(hop, OpOpDG.SEQ)
				&& HopRewriteUtils.hasOnlyUnaryBinaryParents(hop, true))
			|| (HopRewriteUtils.isNary(hop, OpOpN.MIN, OpOpN.MAX, OpOpN.PLUS) && hop.isMatrix())
			|| (HopRewriteUtils.isDnn(hop, OpOpDnn.BIASADD, OpOpDnn.BIASMULT)
				&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown());
	}

	@Override
	public boolean fuse(Hop hop, Hop input) {
		return !isClosed() && (isValidOperation(hop) 
			|| HopRewriteUtils.isAggUnaryOp(hop, SUPPORTED_AGG)
			|| (HopRewriteUtils.isMatrixMultiply(hop)
				&& hop.getDim1()==1 && hop.getDim2()==1)
				&& HopRewriteUtils.isTransposeOperation(hop.getInput().get(0))
			|| (HopRewriteUtils.isTransposeOperation(hop) 
				&& hop.getDim1()==1 && hop.getDim2()>1))
			|| (HopRewriteUtils.isNary(hop, OpOpN.MIN, OpOpN.MAX, OpOpN.PLUS) && hop.isMatrix())
			|| (HopRewriteUtils.isDnn(hop, OpOpDnn.BIASADD, OpOpDnn.BIASMULT)
				&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown());
	}

	@Override
	public boolean merge(Hop hop, Hop input) {
		//merge of other cell tpl possible
		return (!isClosed() && (isValidOperation(hop) 
			|| (hop instanceof AggBinaryOp && hop.getInput().indexOf(input)==0 
				&& HopRewriteUtils.isTransposeOperation(input))))
			|| (HopRewriteUtils.isDataGenOpWithLiteralInputs(input, OpOpDG.SEQ)
				&& HopRewriteUtils.hasOnlyUnaryBinaryParents(input, false))
			|| (HopRewriteUtils.isNary(hop, OpOpN.MIN, OpOpN.MAX, OpOpN.PLUS) && hop.isMatrix())
			|| (HopRewriteUtils.isDnn(hop, OpOpDnn.BIASADD, OpOpDnn.BIASMULT)
				&& hop.getInput().get(0).dimsKnown() && hop.getInput().get(1).dimsKnown());
	}

	@Override
	public CloseType close(Hop hop) {
		//need to close cell tpl after aggregation, see fuse for exact properties
		if( HopRewriteUtils.isAggUnaryOp(hop, SUPPORTED_AGG)
			|| (HopRewriteUtils.isMatrixMultiply(hop) && hop.getDim1()==1 && hop.getDim2()==1) )
			return CloseType.CLOSED_VALID;
		else if( hop instanceof AggUnaryOp || hop instanceof AggBinaryOp )
			return CloseType.CLOSED_INVALID;
		else
			return CloseType.OPEN_VALID;
	}

	@Override
	public Pair<Hop[], CNodeTpl> constructCplan(Hop hop, CPlanMemoTable memo, boolean compileLiterals) 
	{
		//recursively process required cplan output
		HashSet<Hop> inHops = new HashSet<>();
		HashMap<Long, CNode> tmp = new HashMap<>();
		hop.resetVisitStatus();
		rConstructCplan(hop, memo, tmp, inHops, compileLiterals);
		hop.resetVisitStatus();
		
		//reorder inputs (ensure matrices/vectors come first) and prune literals
		//note: we order by number of cells and subsequently sparsity to ensure
		//that sparse inputs are used as the main input w/o unnecessary conversion
		Hop[] sinHops = inHops.stream()
			.filter(h -> !(h.getDataType().isScalar() && tmp.get(h.getHopID()).isLiteral()))
			.sorted(new HopInputComparator()).toArray(Hop[]::new);
		
		//prepare input nodes
		ArrayList<CNode> inputs = new ArrayList<>();
		for( Hop in : sinHops )
			inputs.add(tmp.get(in.getHopID()));
		
		//sanity check for pure scalar inputs
		if( inputs.stream().allMatch(h -> h.getDataType().isScalar()) )
			return null; //later eliminated by cleanupCPlans
		
		//construct template node
		CNode output = tmp.get(hop.getHopID());
		CNodeCell tpl = new CNodeCell(inputs, output);
		tpl.setCellType(TemplateUtils.getCellType(hop));
		tpl.setAggOp(TemplateUtils.getAggOp(hop));
		tpl.setSparseSafe(isSparseSafe(Arrays.asList(hop), sinHops[0], 
			Arrays.asList(tpl.getOutput()), Arrays.asList(tpl.getAggOp()), false));
		tpl.setContainsSeq(rContainsSeq(tpl.getOutput(), new HashSet<>()));
		tpl.setRequiresCastDtm(hop instanceof AggBinaryOp);
		tpl.setBeginLine(hop.getBeginLine());
		
		// return cplan instance
		return new Pair<>(sinHops, tpl);
	}
	
	protected void rConstructCplan(Hop hop, CPlanMemoTable memo, HashMap<Long, CNode> tmp, HashSet<Hop> inHops, boolean compileLiterals) 
	{
		//memoization for common subexpression elimination and to avoid redundant work 
		if( tmp.containsKey(hop.getHopID()) )
			return;
		
		MemoTableEntry me = memo.getBest(hop.getHopID(), TemplateType.CELL);
		
		//recursively process required childs
		if( me!=null && me.type.isIn(TemplateType.ROW, TemplateType.OUTER) ) {
			CNodeData cdata = TemplateUtils.createCNodeData(hop, compileLiterals);	
			tmp.put(hop.getHopID(), cdata);
			inHops.add(hop);
			return;
		}
		for( int i=0; i<hop.getInput().size(); i++ ) {
			Hop c = hop.getInput().get(i);
			if( me!=null && me.isPlanRef(i) && !(c instanceof DataOp)
				&& (me.type!=TemplateType.MAGG || memo.contains(c.getHopID(), TemplateType.CELL)))
				rConstructCplan(c, memo, tmp, inHops, compileLiterals);
			else if( me!=null && (me.type==TemplateType.MAGG || me.type==TemplateType.CELL) 
					&& HopRewriteUtils.isMatrixMultiply(hop) && i==0 ) { //skip transpose
				if( c.getInput().get(0) instanceof DataOp ) {
					tmp.put(c.getInput().get(0).getHopID(),
						TemplateUtils.createCNodeData(c.getInput().get(0), compileLiterals));
					inHops.add(c.getInput().get(0));
				}
				else
					rConstructCplan(c.getInput().get(0), memo, tmp, inHops, compileLiterals);
			}
			else {
				tmp.put(c.getHopID(), TemplateUtils.createCNodeData(c, compileLiterals));
				inHops.add(c);
			}
		}
		
		//construct cnode for current hop
		CNode out = null;
		if(hop instanceof UnaryOp) {
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			
			String primitiveOpName = ((UnaryOp)hop).getOp().name();
			out = new CNodeUnary(cdata1, UnaryType.valueOf(primitiveOpName));
		}
		else if(hop instanceof BinaryOp) {
			BinaryOp bop = (BinaryOp) hop;
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			String primitiveOpName = bop.getOp().name();
			
			//add lookups if required
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			cdata2 = TemplateUtils.wrapLookupIfNecessary(cdata2, hop.getInput().get(1));
			
			//construct binary cnode
			out = new CNodeBinary(cdata1, cdata2, 
				BinType.valueOf(primitiveOpName));
		}
		else if(hop instanceof TernaryOp) {
			TernaryOp top = (TernaryOp) hop;
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			CNode cdata3 = tmp.get(hop.getInput().get(2).getHopID());
			
			//add lookups if required
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			cdata2 = TemplateUtils.wrapLookupIfNecessary(cdata2, hop.getInput().get(1));
			cdata3 = TemplateUtils.wrapLookupIfNecessary(cdata3, hop.getInput().get(2));
			
			//construct ternary cnode, primitive operation derived from OpOp3
			out = new CNodeTernary(cdata1, cdata2, cdata3, 
				TernaryType.valueOf(top.getOp().name()));
		}
		else if( HopRewriteUtils.isDnn(hop, OpOpDnn.BIASADD, OpOpDnn.BIASMULT) ) {
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
			long c = hop.getInput().get(0).getDim2() / hop.getInput().get(1).getDim1();
			CNode cdata3 = TemplateUtils.createCNodeData(new LiteralOp(c), true);
			out = new CNodeTernary(cdata1, cdata2, cdata3,
				TernaryType.valueOf(((DnnOp)hop).getOp().name()));
		}
		else if( HopRewriteUtils.isNary(hop, OpOpN.MIN, OpOpN.MAX, OpOpN.PLUS) ) {
			String op = ((NaryOp)hop).getOp().name();
			CNode[] inputs = hop.getInput().stream().map(c -> 
				TemplateUtils.wrapLookupIfNecessary(tmp.get(c.getHopID()), c)).toArray(CNode[]::new);
			out = new CNodeBinary(inputs[0], inputs[1], BinType.valueOf(op));
			for( int i=2; i<hop.getInput().size(); i++ )
				out = new CNodeBinary(out, inputs[i], BinType.valueOf(op));
		}
		else if( hop instanceof ParameterizedBuiltinOp ) {
			CNode cdata1 = tmp.get(((ParameterizedBuiltinOp)hop).getTargetHop().getHopID());
			cdata1 = TemplateUtils.wrapLookupIfNecessary(cdata1, hop.getInput().get(0));
			
			CNode cdata2 = tmp.get(((ParameterizedBuiltinOp)hop).getParameterHop("pattern").getHopID());
			CNode cdata3 = tmp.get(((ParameterizedBuiltinOp)hop).getParameterHop("replacement").getHopID());
			TernaryType ttype = (cdata2.isLiteral() && cdata2.getVarname().equals("Double.NaN")) ? 
				TernaryType.REPLACE_NAN : TernaryType.REPLACE;
			out = new CNodeTernary(cdata1, cdata2, cdata3, ttype);
		}
		else if( hop instanceof IndexingOp ) {
			CNode cdata1 = tmp.get(hop.getInput().get(0).getHopID());
			out = new CNodeTernary(cdata1, 
				TemplateUtils.createCNodeData(new LiteralOp(hop.getInput().get(0).getDim2()), true), 
				TemplateUtils.createCNodeData(hop.getInput().get(4), true),
				TernaryType.LOOKUP_RC1);
		}
		else if( HopRewriteUtils.isDataGenOp(hop, OpOpDG.SEQ) ) {
			CNodeData from = TemplateUtils.getLiteral(tmp.get(((DataGenOp)hop).getParam(Statement.SEQ_FROM).getHopID()));
			CNodeData to = TemplateUtils.getLiteral(tmp.get(((DataGenOp)hop).getParam(Statement.SEQ_TO).getHopID()));
			CNodeData incr = TemplateUtils.getLiteral(tmp.get(((DataGenOp)hop).getParam(Statement.SEQ_INCR).getHopID()));
			if( Double.parseDouble(from.getVarname()) > Double.parseDouble(to.getVarname())
				&& Double.parseDouble(incr.getVarname()) > 0 ) {
				incr = TemplateUtils.createCNodeData(new LiteralOp("-"+incr.getVarname()), true);
			}
			out = new CNodeBinary(from, incr, BinType.SEQ_RIX);
		}
		else if( HopRewriteUtils.isTransposeOperation(hop) ) {
			out = TemplateUtils.skipTranspose(tmp.get(hop.getHopID()), 
				hop, tmp, compileLiterals);
			//correct indexing types of existing lookups
			if( !HopRewriteUtils.containsOp(hop.getParent(), AggBinaryOp.class) )
				TemplateUtils.rFlipVectorLookups(out);
			//maintain input hops
			if( out instanceof CNodeData && !inHops.contains(hop.getInput().get(0)) )
				inHops.add(hop.getInput().get(0));
		}
		else if( hop instanceof AggUnaryOp ) {
			//aggregation handled in template implementation (note: we do not compile 
			//^2 of SUM_SQ into the operator to simplify the detection of single operators)
			out = tmp.get(hop.getInput().get(0).getHopID());
		}
		else if( hop instanceof AggBinaryOp ) {
			//guaranteed to be a dot product, so there are two cases:
			//(1) t(X)%*%X -> sum(X^2) and t(X) %*% Y -> sum(X*Y)
			if( HopRewriteUtils.isTransposeOfItself(hop.getInput().get(0), hop.getInput().get(1)) ) {
				CNode cdata1 = tmp.get(hop.getInput().get(1).getHopID());
				if( TemplateUtils.isColVector(cdata1) )
					cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
				out = new CNodeUnary(cdata1, UnaryType.POW2);
			}
			else {
				CNode cdata1 = TemplateUtils.skipTranspose(tmp.get(hop.getInput().get(0).getHopID()), 
						hop.getInput().get(0), tmp, compileLiterals);
				if( cdata1 instanceof CNodeData && !inHops.contains(hop.getInput().get(0).getInput().get(0)) )
					inHops.add(hop.getInput().get(0).getInput().get(0));
				if( TemplateUtils.isColVector(cdata1) )
					cdata1 = new CNodeUnary(cdata1, UnaryType.LOOKUP_R);
				CNode cdata2 = tmp.get(hop.getInput().get(1).getHopID());
				if( TemplateUtils.isColVector(cdata2) )
					cdata2 = new CNodeUnary(cdata2, UnaryType.LOOKUP_R);
				out = new CNodeBinary(cdata1, cdata2, BinType.MULT);
			}
		} 
	
		if( out == null ) {
			throw new HopsException(hop.getHopID()+" "+hop.getOpString());
		}
		
		tmp.put(hop.getHopID(), out);
	}
	
	protected static boolean isValidOperation(Hop hop) 
	{
		//prepare indicators for binary operations
		boolean isBinaryMatrixScalar = false;
		boolean isBinaryMatrixVector = false;
		boolean isBinaryMatrixMatrix = false;
		if( hop instanceof BinaryOp && hop.getDataType().isMatrix() && !((BinaryOp)hop).isOuter() ) {
			Hop left = hop.getInput().get(0);
			Hop right = hop.getInput().get(1);
			isBinaryMatrixScalar = (left.getDataType().isScalar() || right.getDataType().isScalar());
			isBinaryMatrixVector = hop.dimsKnown() 
				&& ((left.getDataType().isMatrix() && TemplateUtils.isVectorOrScalar(right)) 
				|| (right.getDataType().isMatrix() && TemplateUtils.isVectorOrScalar(left)) );
			isBinaryMatrixMatrix = hop.dimsKnown() && HopRewriteUtils.isEqualSize(left, right)
				&& left.getDataType().isMatrix() && right.getDataType().isMatrix();
		}
		
		//prepare indicators for ternary operations
		boolean isTernaryVectorScalarVector = false;
		boolean isTernaryMatrixScalarMatrixDense = false;
		boolean isTernaryIfElse = (HopRewriteUtils.isTernary(hop, OpOp3.IFELSE) && hop.getDataType().isMatrix());
		if( hop instanceof TernaryOp && hop.getInput().size()==3 && hop.dimsKnown() 
			&& HopRewriteUtils.checkInputDataTypes(hop, DataType.MATRIX, DataType.SCALAR, DataType.MATRIX) ) {
			Hop left = hop.getInput().get(0);
			Hop right = hop.getInput().get(2);
			isTernaryVectorScalarVector = TemplateUtils.isVector(left) && TemplateUtils.isVector(right);
			isTernaryMatrixScalarMatrixDense = HopRewriteUtils.isEqualSize(left, right) 
				&& !HopRewriteUtils.isSparse(left) && !HopRewriteUtils.isSparse(right);
		}
		
		//check supported unary, binary, ternary operations
		return hop.getDataType() == DataType.MATRIX && TemplateUtils.isOperationSupported(hop) && (hop instanceof UnaryOp 
				|| isBinaryMatrixScalar || isBinaryMatrixVector || isBinaryMatrixMatrix
				|| isTernaryVectorScalarVector || isTernaryMatrixScalarMatrixDense || isTernaryIfElse
				|| (hop instanceof ParameterizedBuiltinOp && ((ParameterizedBuiltinOp)hop).getOp()==ParamBuiltinOp.REPLACE));
	}
	
	protected boolean isSparseSafe(List<Hop> roots, Hop mainInput, List<CNode> outputs, List<AggOp> aggOps, boolean onlySum) {
		boolean ret = true;
		for( int i=0; i<outputs.size() && ret; i++ ) {
			Hop root = (roots.get(i) instanceof AggUnaryOp || roots.get(i) 
				instanceof AggBinaryOp) ? roots.get(i).getInput().get(0) : roots.get(i);
			ret &= (HopRewriteUtils.isBinarySparseSafe(root) 
					&& root.getInput().contains(mainInput))
				|| (HopRewriteUtils.isBinary(root, OpOp2.DIV) 
					&& root.getInput().get(0) == mainInput)
				|| (TemplateUtils.rIsSparseSafeOnly(outputs.get(i), BinType.MULT)
					&& TemplateUtils.rContainsInput(outputs.get(i), mainInput.getHopID()));
			if( onlySum )
				ret &= (aggOps.get(i)==AggOp.SUM || aggOps.get(i)==AggOp.SUM_SQ);
		}
		return ret;
	}
	
	protected boolean rContainsSeq(CNode node, HashSet<Long> memo) {
		if( memo.contains(node.getID()) )
			return false;
		boolean ret = TemplateUtils.isBinary(node, BinType.SEQ_RIX);
		for( CNode c : node.getInput() )
			ret |= rContainsSeq(c, memo);
		memo.add(node.getID());
		return ret;
	}
	
	/**
	 * Comparator to order input hops of the cell template. We try to order 
	 * matrices-vectors-scalars via sorting by number of cells and for 
	 * equal number of cells by sparsity to prefer sparse inputs as the main 
	 * input for sparsity exploitation.
	 */
	public static class HopInputComparator implements Comparator<Hop> 
	{
		private final Hop _driver;
		
		public HopInputComparator() {
			this(null);
		}
		
		public HopInputComparator(Hop driver) {
			_driver = driver;
		}
		
		@Override
		public int compare(Hop h1, Hop h2) {
			long ncells1 = h1.isScalar() ? Long.MIN_VALUE : 
				h1.dimsKnown() ? h1.getLength() : Long.MAX_VALUE;
			long ncells2 = h2.isScalar() ? Long.MIN_VALUE :
				h2.dimsKnown() ? h2.getLength() : Long.MAX_VALUE;
			if( ncells1 > ncells2 || h1 == _driver )
				return -1;
			else if( ncells1 < ncells2 || h2 == _driver)
				return 1;
			if( h1.isScalar() && h2.isScalar() )
				return Long.compare(h1.getHopID(), h2.getHopID());
			return (h1.dimsKnown(true) && h2.dimsKnown(true) && h1.getNnz() != h2.getNnz()
				&& (HopRewriteUtils.isSparse(h1, 1.0) || HopRewriteUtils.isSparse(h2, 1.0))) ?
				Long.compare(h1.getNnz(), h2.getNnz()) :
				Long.compare(h1.getHopID(), h2.getHopID());
		}
	}
}
