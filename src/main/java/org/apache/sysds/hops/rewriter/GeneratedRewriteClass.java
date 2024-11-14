package org.apache.sysds.hops.rewriter;

import java.util.ArrayList;
import java.util.function.Function;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;

public class GeneratedRewriteClass implements Function {

	@Override
	public Object apply( Object hi ) {
		if ( hi == null )
			return null;

		hi = _applyRewrite0((Hop) hi);
		hi = _applyRewrite1((Hop) hi);
		hi = _applyRewrite2((Hop) hi);
		hi = _applyRewrite3((Hop) hi);
		hi = _applyRewrite4((Hop) hi);
		hi = _applyRewrite5((Hop) hi);
		hi = _applyRewrite6((Hop) hi);
		hi = _applyRewrite7((Hop) hi);
		hi = _applyRewrite8((Hop) hi);
		hi = _applyRewrite9((Hop) hi);
		hi = _applyRewrite10((Hop) hi);
		hi = _applyRewrite11((Hop) hi);
		hi = _applyRewrite12((Hop) hi);
		hi = _applyRewrite13((Hop) hi);
		hi = _applyRewrite14((Hop) hi);
		hi = _applyRewrite15((Hop) hi);
		hi = _applyRewrite17((Hop) hi);
		hi = _applyRewrite18((Hop) hi);
		hi = _applyRewrite19((Hop) hi);
		hi = _applyRewrite20((Hop) hi);
		hi = _applyRewrite21((Hop) hi);
		hi = _applyRewrite22((Hop) hi);
		hi = _applyRewrite23((Hop) hi);
		hi = _applyRewrite24((Hop) hi);
		hi = _applyRewrite25((Hop) hi);
		hi = _applyRewrite26((Hop) hi);
		hi = _applyRewrite27((Hop) hi);
		hi = _applyRewrite28((Hop) hi);
		hi = _applyRewrite30((Hop) hi);
		hi = _applyRewrite31((Hop) hi);
		hi = _applyRewrite45((Hop) hi);
		hi = _applyRewrite47((Hop) hi);
		hi = _applyRewrite48((Hop) hi);
		hi = _applyRewrite49((Hop) hi);
		hi = _applyRewrite52((Hop) hi);
		hi = _applyRewrite53((Hop) hi);
		hi = _applyRewrite54((Hop) hi);
		hi = _applyRewrite55((Hop) hi);
		hi = _applyRewrite56((Hop) hi);
		hi = _applyRewrite57((Hop) hi);
		hi = _applyRewrite58((Hop) hi);
		hi = _applyRewrite59((Hop) hi);
		hi = _applyRewrite63((Hop) hi);
		hi = _applyRewrite64((Hop) hi);
		hi = _applyRewrite65((Hop) hi);
		hi = _applyRewrite66((Hop) hi);
		hi = _applyRewrite67((Hop) hi);
		hi = _applyRewrite68((Hop) hi);
		hi = _applyRewrite69((Hop) hi);
		hi = _applyRewrite70((Hop) hi);
		hi = _applyRewrite71((Hop) hi);
		hi = _applyRewrite72((Hop) hi);
		hi = _applyRewrite73((Hop) hi);
		hi = _applyRewrite74((Hop) hi);
		hi = _applyRewrite75((Hop) hi);
		hi = _applyRewrite77((Hop) hi);
		hi = _applyRewrite79((Hop) hi);
		hi = _applyRewrite80((Hop) hi);
		hi = _applyRewrite84((Hop) hi);
		hi = _applyRewrite85((Hop) hi);
		hi = _applyRewrite86((Hop) hi);
		hi = _applyRewrite87((Hop) hi);
		hi = _applyRewrite88((Hop) hi);
		hi = _applyRewrite89((Hop) hi);
		hi = _applyRewrite90((Hop) hi);
		hi = _applyRewrite96((Hop) hi);
		return hi;
	}

	// Implementation of the rule t(t(Z)) <=> Z
	private static Hop _applyRewrite0(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(t(Z)) <=> Z");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_0_0;
	}

	// Implementation of the rule +(Z,0) <=> Z
	private static Hop _applyRewrite1(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(Z,0) <=> Z");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule +(0.0,Z) <=> Z
	private static Hop _applyRewrite2(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(0.0,Z) <=> Z");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule +(+(-(max_iteration,1),1),0) <=> max_iteration
	private static Hop _applyRewrite3(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.SCALAR || (c_hi_0_0.getValueType() != Types.ValueType.INT64 && c_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0_1 = (LiteralOp) hi_0_0_1;

		if ( l_hi_0_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_0_1 != hi_0_1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(+(-(max_iteration,1),1),0) <=> max_iteration");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0_0_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0_0_0;
	}

	// Implementation of the rule *(1,+(-(max_iteration,1),1)) <=> max_iteration
	private static Hop _applyRewrite4(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.INT64 && l_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.INT64 && c_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.SCALAR || (c_hi_1_0.getValueType() != Types.ValueType.INT64 && c_hi_1_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_0 != hi_1_0_1 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_0 != hi_1_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(1,+(-(max_iteration,1),1)) <=> max_iteration");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1_0_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return hi_1_0_0;
	}

	// Implementation of the rule *(1,max_iteration) <=> max_iteration
	private static Hop _applyRewrite5(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.INT64 && l_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(1,max_iteration) <=> max_iteration");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule -(+(max_iteration,1),1) <=> max_iteration
	private static Hop _applyRewrite6(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(max_iteration,1),1) <=> max_iteration");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return hi_0_0;
	}

	// Implementation of the rule +(max_iteration,0.0) <=> max_iteration
	private static Hop _applyRewrite7(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(max_iteration,0.0) <=> max_iteration");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule +(max_iteration,0) <=> max_iteration
	private static Hop _applyRewrite8(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(max_iteration,0) <=> max_iteration");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule +(-(max_iteration,1),1) <=> max_iteration
	private static Hop _applyRewrite9(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(max_iteration,1),1) <=> max_iteration");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return hi_0_0;
	}

	// Implementation of the rule *(9999,/(parsertemp2,9999.0)) <=> parsertemp2
	private static Hop _applyRewrite10(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.INT64 && l_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0.getLongValue() != 9999 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_1_1.getValueType() != Types.ValueType.FP64 && l_hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_1.getDoubleValue() != 9999.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(9999,/(parsertemp2,9999.0)) <=> parsertemp2");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return hi_1_0;
	}

	// Implementation of the rule /(parsertemp2,1) <=> parsertemp2
	private static Hop _applyRewrite11(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(parsertemp2,1) <=> parsertemp2");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule *(1,parsertemp2) <=> parsertemp2
	private static Hop _applyRewrite12(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.INT64 && l_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(1,parsertemp2) <=> parsertemp2");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule *(999,/(parsertemp2,999.0)) <=> parsertemp2
	private static Hop _applyRewrite13(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.INT64 && l_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0.getLongValue() != 999 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_1_1.getValueType() != Types.ValueType.FP64 && l_hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_1.getDoubleValue() != 999.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(999,/(parsertemp2,999.0)) <=> parsertemp2");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return hi_1_0;
	}

	// Implementation of the rule *(1.0,parsertemp2) <=> parsertemp2
	private static Hop _applyRewrite14(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.FP64 && l_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(1.0,parsertemp2) <=> parsertemp2");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule +(%*%(is_LT_infinite,flip_pos),%*%(a606825c-603f-4984-b8ad-2746fe527275,flip_pos)) <=> %*%(+(a606825c-603f-4984-b8ad-2746fe527275,is_LT_infinite),flip_pos)
	private static Hop _applyRewrite15(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		Hop hi_1 = hi.getInput(1);

		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_0_1 != hi_1_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(%*%(is_LT_infinite,flip_pos),%*%(a606825c-603f-4984-b8ad-2746fe527275,flip_pos)) <=> %*%(+(a606825c-603f-4984-b8ad-2746fe527275,is_LT_infinite),flip_pos)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_0, Types.OpOp2.PLUS);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(0,-(parsertemp138264,R)) <=> -(R,parsertemp138264)
	private static Hop _applyRewrite17(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR || (l_hi_0.getValueType() != Types.ValueType.INT64 && l_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0.getLongValue() != 0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		Hop hi_1_1 = hi_1.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(0,-(parsertemp138264,R)) <=> -(R,parsertemp138264)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v1;
	}

	// Implementation of the rule rowSums(t(2aff7584-58ef-4f60-93d6-d06940408113)) <=> t(colSums(2aff7584-58ef-4f60-93d6-d06940408113))
	private static Hop _applyRewrite18(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(t(2aff7584-58ef-4f60-93d6-d06940408113)) <=> t(colSums(2aff7584-58ef-4f60-93d6-d06940408113))");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule t(+(f0b918f3-db49-485e-9313-c34574ce5ac4,t(92d709d1-36f3-4fc4-b0e1-b620dd26ca75))) <=> +(92d709d1-36f3-4fc4-b0e1-b620dd26ca75,t(f0b918f3-db49-485e-9313-c34574ce5ac4))
	private static Hop _applyRewrite19(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(f0b918f3-db49-485e-9313-c34574ce5ac4,t(92d709d1-36f3-4fc4-b0e1-b620dd26ca75))) <=> +(92d709d1-36f3-4fc4-b0e1-b620dd26ca75,t(f0b918f3-db49-485e-9313-c34574ce5ac4))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule /(scale_lambda,1000) <=> *(scale_lambda,0.001)
	private static Hop _applyRewrite20(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1000 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(scale_lambda,1000) <=> *(scale_lambda,0.001)");
		LiteralOp l1 = new LiteralOp( 0.001 );
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, l1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(817fad0c-97d8-4a72-90f5-93ef5890ee9a,100000) <=> *(817fad0c-97d8-4a72-90f5-93ef5890ee9a,1.0E-5)
	private static Hop _applyRewrite21(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 100000 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(817fad0c-97d8-4a72-90f5-93ef5890ee9a,100000) <=> *(817fad0c-97d8-4a72-90f5-93ef5890ee9a,1.0E-5)");
		LiteralOp l1 = new LiteralOp( 1.0E-5 );
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, l1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(parsertemp73612,10000) <=> *(parsertemp73612,1.0E-4)
	private static Hop _applyRewrite22(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 10000 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(parsertemp73612,10000) <=> *(parsertemp73612,1.0E-4)");
		LiteralOp l1 = new LiteralOp( 1.0E-4 );
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, l1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(parsertemp6002,0.5) <=> *(2.0,parsertemp6002)
	private static Hop _applyRewrite23(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.5 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(parsertemp6002,0.5) <=> *(2.0,parsertemp6002)");
		LiteralOp l1 = new LiteralOp( 2.0 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(input,100) <=> *(0.01,input)
	private static Hop _applyRewrite24(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 100 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(input,100) <=> *(0.01,input)");
		LiteralOp l1 = new LiteralOp( 0.01 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(de36e8e9-da90-450f-b9be-8c2def7326a4,2) <=> *(0.5,de36e8e9-da90-450f-b9be-8c2def7326a4)
	private static Hop _applyRewrite25(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 2 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(de36e8e9-da90-450f-b9be-8c2def7326a4,2) <=> *(0.5,de36e8e9-da90-450f-b9be-8c2def7326a4)");
		LiteralOp l1 = new LiteralOp( 0.5 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(de36e8e9-da90-450f-b9be-8c2def7326a4,2.0) <=> *(0.5,de36e8e9-da90-450f-b9be-8c2def7326a4)
	private static Hop _applyRewrite26(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 2.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(de36e8e9-da90-450f-b9be-8c2def7326a4,2.0) <=> *(0.5,de36e8e9-da90-450f-b9be-8c2def7326a4)");
		LiteralOp l1 = new LiteralOp( 0.5 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(t(parsertemp200774),t(parsertemp200777)) <=> t(/(parsertemp200774,parsertemp200777))
	private static Hop _applyRewrite27(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(t(parsertemp200774),t(parsertemp200777)) <=> t(/(parsertemp200774,parsertemp200777))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule t(/(7a9d8263-acab-419a-bb15-b5252e4cf52c,t(0bce686a-3c63-4693-80cc-babaea0e2d38))) <=> /(t(7a9d8263-acab-419a-bb15-b5252e4cf52c),0bce686a-3c63-4693-80cc-babaea0e2d38)
	private static Hop _applyRewrite28(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(7a9d8263-acab-419a-bb15-b5252e4cf52c,t(0bce686a-3c63-4693-80cc-babaea0e2d38))) <=> /(t(7a9d8263-acab-419a-bb15-b5252e4cf52c),0bce686a-3c63-4693-80cc-babaea0e2d38)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule /(*(weight,t(12b1099a-02ee-4f1d-b79d-776cdbf10b16)),t(9c2c85cd-a4c7-4ef0-bbe3-3c0df24720ac)) <=> *(t(/(12b1099a-02ee-4f1d-b79d-776cdbf10b16,9c2c85cd-a4c7-4ef0-bbe3-3c0df24720ac)),weight)
	private static Hop _applyRewrite30(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(weight,t(12b1099a-02ee-4f1d-b79d-776cdbf10b16)),t(9c2c85cd-a4c7-4ef0-bbe3-3c0df24720ac)) <=> *(t(/(12b1099a-02ee-4f1d-b79d-776cdbf10b16,9c2c85cd-a4c7-4ef0-bbe3-3c0df24720ac)),weight)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1_0, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule t(/(*(12b1099a-02ee-4f1d-b79d-776cdbf10b16,t(weight)),9c2c85cd-a4c7-4ef0-bbe3-3c0df24720ac)) <=> *(t(/(12b1099a-02ee-4f1d-b79d-776cdbf10b16,9c2c85cd-a4c7-4ef0-bbe3-3c0df24720ac)),weight)
	private static Hop _applyRewrite31(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0_1 = (ReorgOp) hi_0_0_1;

		if ( c_hi_0_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1_0 = hi_0_0_1.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(*(12b1099a-02ee-4f1d-b79d-776cdbf10b16,t(weight)),9c2c85cd-a4c7-4ef0-bbe3-3c0df24720ac)) <=> *(t(/(12b1099a-02ee-4f1d-b79d-776cdbf10b16,9c2c85cd-a4c7-4ef0-bbe3-3c0df24720ac)),weight)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0_1_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);

		return v3;
	}

	// Implementation of the rule t(/(%*%(t(V),W),t(parsertemp63810))) <=> /(%*%(t(W),V),parsertemp63810)
	private static Hop _applyRewrite45(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !HopRewriteUtils.isMatrixMultiply(hi_0_0) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( !(hi_0_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0_0 = (ReorgOp) hi_0_0_0;

		if ( c_hi_0_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0_0 = hi_0_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(%*%(t(V),W),t(parsertemp63810))) <=> /(%*%(t(W),V),parsertemp63810)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0_1);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule %*%(t(y),t(parsertemp22849)) <=> t(%*%(parsertemp22849,y))
	private static Hop _applyRewrite47(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(t(y),t(parsertemp22849)) <=> t(%*%(parsertemp22849,y))");
		AggBinaryOp v1 = HopRewriteUtils.createMatrixMultiply(hi_1_0, hi_0_0);
		ReorgOp v2 = HopRewriteUtils.createTranspose(v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule t(%*%(t(b4658a47-8370-4d5d-a8e7-3c4a9dd54933),p)) <=> %*%(t(p),b4658a47-8370-4d5d-a8e7-3c4a9dd54933)
	private static Hop _applyRewrite48(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(%*%(t(b4658a47-8370-4d5d-a8e7-3c4a9dd54933),p)) <=> %*%(t(p),b4658a47-8370-4d5d-a8e7-3c4a9dd54933)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_0_0);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(%*%(fdf26f6e-c887-4522-ab95-dbdafa92a825,t(X))) <=> %*%(X,t(fdf26f6e-c887-4522-ab95-dbdafa92a825))
	private static Hop _applyRewrite49(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(%*%(fdf26f6e-c887-4522-ab95-dbdafa92a825,t(X))) <=> %*%(X,t(fdf26f6e-c887-4522-ab95-dbdafa92a825))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(hi_0_1_0, v1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule +(-(4264f8bc-a021-4a16-8791-12f9f85cff6e,+(20e772fb-a9e0-4eb0-8e27-9526cd3a59f8,1)),1) <=> -(4264f8bc-a021-4a16-8791-12f9f85cff6e,20e772fb-a9e0-4eb0-8e27-9526cd3a59f8)
	private static Hop _applyRewrite52(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.SCALAR || (c_hi_0_1.getValueType() != Types.ValueType.INT64 && c_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( !(hi_0_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1_1 = (LiteralOp) hi_0_1_1;

		if ( l_hi_0_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1_1.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(4264f8bc-a021-4a16-8791-12f9f85cff6e,+(20e772fb-a9e0-4eb0-8e27-9526cd3a59f8,1)),1) <=> -(4264f8bc-a021-4a16-8791-12f9f85cff6e,20e772fb-a9e0-4eb0-8e27-9526cd3a59f8)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1_1);

		return v1;
	}

	// Implementation of the rule -(+(+(258ce525-6761-4542-bf56-32aba43e914e,1),22d96f18-66a7-4b82-be2e-450a03fb0961),1) <=> +(258ce525-6761-4542-bf56-32aba43e914e,22d96f18-66a7-4b82-be2e-450a03fb0961)
	private static Hop _applyRewrite53(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.SCALAR || (c_hi_0_0.getValueType() != Types.ValueType.INT64 && c_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0_1 = (LiteralOp) hi_0_0_1;

		if ( l_hi_0_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(+(258ce525-6761-4542-bf56-32aba43e914e,1),22d96f18-66a7-4b82-be2e-450a03fb0961),1) <=> +(258ce525-6761-4542-bf56-32aba43e914e,22d96f18-66a7-4b82-be2e-450a03fb0961)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);

		return v1;
	}

	// Implementation of the rule +(*(-(d18e4f8d-04ff-4b5e-ad7a-66fad8fbb447,1),11),11) <=> *(d18e4f8d-04ff-4b5e-ad7a-66fad8fbb447,11)
	private static Hop _applyRewrite54(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.SCALAR || (c_hi_0_0.getValueType() != Types.ValueType.INT64 && c_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0_1 = (LiteralOp) hi_0_0_1;

		if ( l_hi_0_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 11 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(-(d18e4f8d-04ff-4b5e-ad7a-66fad8fbb447,1),11),11) <=> *(d18e4f8d-04ff-4b5e-ad7a-66fad8fbb447,11)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);

		return v1;
	}

	// Implementation of the rule +(*(-(f8e4c38e-335b-47d6-ab40-6ad7ba1a8cb6,1),12),12) <=> *(f8e4c38e-335b-47d6-ab40-6ad7ba1a8cb6,12)
	private static Hop _applyRewrite55(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.SCALAR || (c_hi_0_0.getValueType() != Types.ValueType.INT64 && c_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0_1 = (LiteralOp) hi_0_0_1;

		if ( l_hi_0_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 12 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(-(f8e4c38e-335b-47d6-ab40-6ad7ba1a8cb6,1),12),12) <=> *(f8e4c38e-335b-47d6-ab40-6ad7ba1a8cb6,12)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);

		return v1;
	}

	// Implementation of the rule +(*(-(161b6a14-aa26-44e0-9988-8a102d2b1505,1),61),61) <=> *(161b6a14-aa26-44e0-9988-8a102d2b1505,61)
	private static Hop _applyRewrite56(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.SCALAR || (c_hi_0_0.getValueType() != Types.ValueType.INT64 && c_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0_1 = (LiteralOp) hi_0_0_1;

		if ( l_hi_0_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 61 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(-(161b6a14-aa26-44e0-9988-8a102d2b1505,1),61),61) <=> *(161b6a14-aa26-44e0-9988-8a102d2b1505,61)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);

		return v1;
	}

	// Implementation of the rule +(*(3,-(sample_block_size,1)),3) <=> *(sample_block_size,3)
	private static Hop _applyRewrite57(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.INT64 && l_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0.getLongValue() != 3 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.SCALAR || (c_hi_0_1.getValueType() != Types.ValueType.INT64 && c_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( !(hi_0_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1_1 = (LiteralOp) hi_0_1_1;

		if ( l_hi_0_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1_1.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_0 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(3,-(sample_block_size,1)),3) <=> *(sample_block_size,3)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1_1);

		return v1;
	}

	// Implementation of the rule +(*(-(IDleft,1),2),2) <=> *(2,IDleft)
	private static Hop _applyRewrite58(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.SCALAR || (c_hi_0_0.getValueType() != Types.ValueType.INT64 && c_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0_1 = (LiteralOp) hi_0_0_1;

		if ( l_hi_0_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 2 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(-(IDleft,1),2),2) <=> *(2,IDleft)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);

		return v1;
	}

	// Implementation of the rule +(*(-(a7eafc09-1bc9-468f-841c-018717e3516f,1),128),128) <=> *(a7eafc09-1bc9-468f-841c-018717e3516f,128)
	private static Hop _applyRewrite59(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.SCALAR || (c_hi_0_0.getValueType() != Types.ValueType.INT64 && c_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0_1 = (LiteralOp) hi_0_0_1;

		if ( l_hi_0_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 128 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(-(a7eafc09-1bc9-468f-841c-018717e3516f,1),128),128) <=> *(a7eafc09-1bc9-468f-841c-018717e3516f,128)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);

		return v1;
	}

	// Implementation of the rule +(+(num_func_invoc,2),3) <=> +(num_func_invoc,5)
	private static Hop _applyRewrite63(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 2 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 3 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(+(num_func_invoc,2),3) <=> +(num_func_invoc,5)");
		LiteralOp l1 = new LiteralOp( 5 );
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, l1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(3,j),1) <=> -(4,j)
	private static Hop _applyRewrite64(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.INT64 && l_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0.getLongValue() != 3 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(3,j),1) <=> -(4,j)");
		LiteralOp l1 = new LiteralOp( 4 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(11,idx),1) <=> -(12,idx)
	private static Hop _applyRewrite65(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.INT64 && l_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0.getLongValue() != 11 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(11,idx),1) <=> -(12,idx)");
		LiteralOp l1 = new LiteralOp( 12 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(+(i6,5),1) <=> +(4,i6)
	private static Hop _applyRewrite66(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 5 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(i6,5),1) <=> +(4,i6)");
		LiteralOp l1 = new LiteralOp( 4 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0_0, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(+(n_group_cols,2),1) <=> +(3,n_group_cols)
	private static Hop _applyRewrite67(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 2 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(+(n_group_cols,2),1) <=> +(3,n_group_cols)");
		LiteralOp l1 = new LiteralOp( 3 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0_0, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(+(n_group_cols,4),1) <=> +(3,n_group_cols)
	private static Hop _applyRewrite68(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 4 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(n_group_cols,4),1) <=> +(3,n_group_cols)");
		LiteralOp l1 = new LiteralOp( 3 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0_0, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(i,3),1) <=> -(i,2)
	private static Hop _applyRewrite69(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 3 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(i,3),1) <=> -(i,2)");
		LiteralOp l1 = new LiteralOp( 2 );
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, l1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(i,4),2) <=> -(i,2)
	private static Hop _applyRewrite70(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 4 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 2 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(i,4),2) <=> -(i,2)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule -(+(00df609e-aade-40bd-a064-55f8c86e920c,0.0),3cd2e2c2-d28c-4736-a9f6-fe856a8ab566) <=> -(00df609e-aade-40bd-a064-55f8c86e920c,3cd2e2c2-d28c-4736-a9f6-fe856a8ab566)
	private static Hop _applyRewrite71(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.FP64 && c_hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.FP64 && l_hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_0_1.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(00df609e-aade-40bd-a064-55f8c86e920c,0.0),3cd2e2c2-d28c-4736-a9f6-fe856a8ab566) <=> -(00df609e-aade-40bd-a064-55f8c86e920c,3cd2e2c2-d28c-4736-a9f6-fe856a8ab566)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule -(+(i,12),1) <=> +(i,11)
	private static Hop _applyRewrite72(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 12 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(i,12),1) <=> +(i,11)");
		LiteralOp l1 = new LiteralOp( 11 );
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, l1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(+(3,i),1) <=> +(2,i)
	private static Hop _applyRewrite73(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.INT64 && l_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0.getLongValue() != 3 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(3,i),1) <=> +(2,i)");
		LiteralOp l1 = new LiteralOp( 2 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0_1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(+(da0b95ae-2a88-4be4-8abf-bcd6f8dac0ed,1),2) <=> -(da0b95ae-2a88-4be4-8abf-bcd6f8dac0ed,1)
	private static Hop _applyRewrite74(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 2 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(da0b95ae-2a88-4be4-8abf-bcd6f8dac0ed,1),2) <=> -(da0b95ae-2a88-4be4-8abf-bcd6f8dac0ed,1)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v1;
	}

	// Implementation of the rule +(-(da0b95ae-2a88-4be4-8abf-bcd6f8dac0ed,2),1) <=> -(da0b95ae-2a88-4be4-8abf-bcd6f8dac0ed,1)
	private static Hop _applyRewrite75(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1 = (LiteralOp) hi_0_1;

		if ( l_hi_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1.getLongValue() != 2 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(da0b95ae-2a88-4be4-8abf-bcd6f8dac0ed,2),1) <=> -(da0b95ae-2a88-4be4-8abf-bcd6f8dac0ed,1)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule /(c111d635-4678-4f5e-8f29-65f5f00a178c,+(983b8ccd-49a7-4e7d-8e25-2ef83ac02f7c,0.0)) <=> /(c111d635-4678-4f5e-8f29-65f5f00a178c,983b8ccd-49a7-4e7d-8e25-2ef83ac02f7c)
	private static Hop _applyRewrite77(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.FP64 && c_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_1 = (LiteralOp) hi_1_1;

		if ( l_hi_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_1_1.getValueType() != Types.ValueType.FP64 && l_hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(c111d635-4678-4f5e-8f29-65f5f00a178c,+(983b8ccd-49a7-4e7d-8e25-2ef83ac02f7c,0.0)) <=> /(c111d635-4678-4f5e-8f29-65f5f00a178c,983b8ccd-49a7-4e7d-8e25-2ef83ac02f7c)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v1;
	}

	// Implementation of the rule +(-(n,-(dba024e5-4ca9-4a04-b066-29c7207ceb49,1)),1) <=> +(-(n,dba024e5-4ca9-4a04-b066-29c7207ceb49),2)
	private static Hop _applyRewrite79(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.SCALAR || (c_hi_0_1.getValueType() != Types.ValueType.INT64 && c_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( !(hi_0_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1_1 = (LiteralOp) hi_0_1_1;

		if ( l_hi_0_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1_1.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_0_1_1 != hi_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(n,-(dba024e5-4ca9-4a04-b066-29c7207ceb49,1)),1) <=> +(-(n,dba024e5-4ca9-4a04-b066-29c7207ceb49),2)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);
		LiteralOp l2 = new LiteralOp( 2 );
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, l2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1_1);

		return v3;
	}

	// Implementation of the rule +(*(3,-(sample_block_size,1)),2) <=> -(*(sample_block_size,3),1)
	private static Hop _applyRewrite80(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR || (l_hi_0_0.getValueType() != Types.ValueType.INT64 && l_hi_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_0.getLongValue() != 3 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.SCALAR || (c_hi_0_1.getValueType() != Types.ValueType.INT64 && c_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( !(hi_0_1_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1_1 = (LiteralOp) hi_0_1_1;

		if ( l_hi_0_1_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1_1.getLongValue() != 1 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 2 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(*(3,-(sample_block_size,1)),2) <=> -(*(sample_block_size,3),1)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(n,-(+(i,12),1a69441b-e4c5-4360-a065-3a9a30a6a883)),1) <=> -(n,-(+(i,11),1a69441b-e4c5-4360-a065-3a9a30a6a883))
	private static Hop _applyRewrite84(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.INT64 && c_hi.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.SCALAR || (c_hi_0.getValueType() != Types.ValueType.INT64 && c_hi_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.SCALAR || (c_hi_0_1.getValueType() != Types.ValueType.INT64 && c_hi_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( !(hi_0_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1_0 = (BinaryOp) hi_0_1_0;

		if ( c_hi_0_1_0.getOp() != Types.OpOp2.PLUS || c_hi_0_1_0.getDataType() != Types.DataType.SCALAR || (c_hi_0_1_0.getValueType() != Types.ValueType.INT64 && c_hi_0_1_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_0_1_0_0 = hi_0_1_0.getInput(0);

		Hop hi_0_1_0_1 = hi_0_1_0.getInput(1);

		if ( !(hi_0_1_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_1_0_1 = (LiteralOp) hi_0_1_0_1;

		if ( l_hi_0_1_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_0_1_0_1.getValueType() != Types.ValueType.INT64 && l_hi_0_1_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_0_1_0_1.getLongValue() != 12 )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(n,-(+(i,12),1a69441b-e4c5-4360-a065-3a9a30a6a883)),1) <=> -(n,-(+(i,11),1a69441b-e4c5-4360-a065-3a9a30a6a883))");
		LiteralOp l1 = new LiteralOp( 11 );
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0_0, l1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_1, Types.OpOp2.MINUS);
		BinaryOp v4 = HopRewriteUtils.createBinary(hi_0_0, v3, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v4);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v4;
	}

	// Implementation of the rule /(2872c6aa-ee7b-4f54-b4e4-c1caeadcfce5,100) <=> *(0.01,2872c6aa-ee7b-4f54-b4e4-c1caeadcfce5)
	private static Hop _applyRewrite85(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 100 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(2872c6aa-ee7b-4f54-b4e4-c1caeadcfce5,100) <=> *(0.01,2872c6aa-ee7b-4f54-b4e4-c1caeadcfce5)");
		LiteralOp l1 = new LiteralOp( 0.01 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(norm_Grad_initial,10000) <=> *(1.0E-4,norm_Grad_initial)
	private static Hop _applyRewrite86(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 10000 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(norm_Grad_initial,10000) <=> *(1.0E-4,norm_Grad_initial)");
		LiteralOp l1 = new LiteralOp( 1.0E-4 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(norm_Grad,10) <=> *(0.1,norm_Grad)
	private static Hop _applyRewrite87(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 10 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(norm_Grad,10) <=> *(0.1,norm_Grad)");
		LiteralOp l1 = new LiteralOp( 0.1 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(AIC_best_orig,1000) <=> *(0.001,AIC_best_orig)
	private static Hop _applyRewrite88(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1000 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(AIC_best_orig,1000) <=> *(0.001,AIC_best_orig)");
		LiteralOp l1 = new LiteralOp( 0.001 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(delta,2.0) <=> *(0.5,delta)
	private static Hop _applyRewrite89(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 2.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(delta,2.0) <=> *(0.5,delta)");
		LiteralOp l1 = new LiteralOp( 0.5 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(delta,2) <=> *(0.5,delta)
	private static Hop _applyRewrite90(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.SCALAR || (c_hi.getValueType() != Types.ValueType.FP64 && c_hi.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 2 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(delta,2) <=> *(0.5,delta)");
		LiteralOp l1 = new LiteralOp( 0.5 );
		BinaryOp v2 = HopRewriteUtils.createBinary(l1, hi_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(e5c50b1b-6325-4972-8589-b987fe7e6ebd,-(0,ccfc5489-b00d-4e1e-abc2-9ef9c3531e19)) <=> -(e5c50b1b-6325-4972-8589-b987fe7e6ebd,ccfc5489-b00d-4e1e-abc2-9ef9c3531e19)
	private static Hop _applyRewrite96(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.SCALAR || (c_hi_1.getValueType() != Types.ValueType.INT64 && c_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR || (l_hi_1_0.getValueType() != Types.ValueType.INT64 && l_hi_1_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1_0.getLongValue() != 0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(e5c50b1b-6325-4972-8589-b987fe7e6ebd,-(0,ccfc5489-b00d-4e1e-abc2-9ef9c3531e19)) <=> -(e5c50b1b-6325-4972-8589-b987fe7e6ebd,ccfc5489-b00d-4e1e-abc2-9ef9c3531e19)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v1;
	}
}
