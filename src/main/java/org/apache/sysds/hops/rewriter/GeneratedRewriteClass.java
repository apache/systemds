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
		hi = _applyRewrite16((Hop) hi);
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
		hi = _applyRewrite29((Hop) hi);
		hi = _applyRewrite30((Hop) hi);
		hi = _applyRewrite31((Hop) hi);
		hi = _applyRewrite32((Hop) hi);
		hi = _applyRewrite33((Hop) hi);
		hi = _applyRewrite34((Hop) hi);
		hi = _applyRewrite35((Hop) hi);
		hi = _applyRewrite36((Hop) hi);
		hi = _applyRewrite37((Hop) hi);
		hi = _applyRewrite38((Hop) hi);
		hi = _applyRewrite39((Hop) hi);
		hi = _applyRewrite40((Hop) hi);
		hi = _applyRewrite41((Hop) hi);
		hi = _applyRewrite42((Hop) hi);
		hi = _applyRewrite43((Hop) hi);
		hi = _applyRewrite44((Hop) hi);
		hi = _applyRewrite45((Hop) hi);
		hi = _applyRewrite46((Hop) hi);
		hi = _applyRewrite47((Hop) hi);
		hi = _applyRewrite48((Hop) hi);
		hi = _applyRewrite49((Hop) hi);
		hi = _applyRewrite50((Hop) hi);
		hi = _applyRewrite51((Hop) hi);
		hi = _applyRewrite52((Hop) hi);
		hi = _applyRewrite53((Hop) hi);
		hi = _applyRewrite54((Hop) hi);
		hi = _applyRewrite55((Hop) hi);
		hi = _applyRewrite56((Hop) hi);
		hi = _applyRewrite57((Hop) hi);
		hi = _applyRewrite58((Hop) hi);
		hi = _applyRewrite59((Hop) hi);
		hi = _applyRewrite60((Hop) hi);
		hi = _applyRewrite61((Hop) hi);
		hi = _applyRewrite62((Hop) hi);
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
		hi = _applyRewrite76((Hop) hi);
		hi = _applyRewrite77((Hop) hi);
		hi = _applyRewrite78((Hop) hi);
		hi = _applyRewrite79((Hop) hi);
		hi = _applyRewrite80((Hop) hi);
		hi = _applyRewrite81((Hop) hi);
		hi = _applyRewrite82((Hop) hi);
		hi = _applyRewrite83((Hop) hi);
		hi = _applyRewrite84((Hop) hi);
		hi = _applyRewrite85((Hop) hi);
		hi = _applyRewrite86((Hop) hi);
		hi = _applyRewrite87((Hop) hi);
		hi = _applyRewrite88((Hop) hi);
		hi = _applyRewrite89((Hop) hi);
		hi = _applyRewrite90((Hop) hi);
		hi = _applyRewrite91((Hop) hi);
		hi = _applyRewrite92((Hop) hi);
		hi = _applyRewrite93((Hop) hi);
		hi = _applyRewrite94((Hop) hi);
		hi = _applyRewrite95((Hop) hi);
		hi = _applyRewrite96((Hop) hi);
		hi = _applyRewrite97((Hop) hi);
		hi = _applyRewrite98((Hop) hi);
		hi = _applyRewrite99((Hop) hi);
		hi = _applyRewrite100((Hop) hi);
		hi = _applyRewrite101((Hop) hi);
		hi = _applyRewrite102((Hop) hi);
		hi = _applyRewrite103((Hop) hi);
		hi = _applyRewrite104((Hop) hi);
		hi = _applyRewrite105((Hop) hi);
		hi = _applyRewrite106((Hop) hi);
		hi = _applyRewrite107((Hop) hi);
		hi = _applyRewrite108((Hop) hi);
		hi = _applyRewrite109((Hop) hi);
		hi = _applyRewrite110((Hop) hi);
		hi = _applyRewrite111((Hop) hi);
		hi = _applyRewrite112((Hop) hi);
		hi = _applyRewrite113((Hop) hi);
		hi = _applyRewrite114((Hop) hi);
		hi = _applyRewrite115((Hop) hi);
		hi = _applyRewrite116((Hop) hi);
		hi = _applyRewrite117((Hop) hi);
		hi = _applyRewrite118((Hop) hi);
		hi = _applyRewrite119((Hop) hi);
		hi = _applyRewrite120((Hop) hi);
		hi = _applyRewrite121((Hop) hi);
		hi = _applyRewrite122((Hop) hi);
		hi = _applyRewrite123((Hop) hi);
		hi = _applyRewrite124((Hop) hi);
		hi = _applyRewrite125((Hop) hi);
		hi = _applyRewrite126((Hop) hi);
		hi = _applyRewrite127((Hop) hi);
		hi = _applyRewrite128((Hop) hi);
		hi = _applyRewrite129((Hop) hi);
		hi = _applyRewrite130((Hop) hi);
		hi = _applyRewrite131((Hop) hi);
		hi = _applyRewrite132((Hop) hi);
		hi = _applyRewrite133((Hop) hi);
		hi = _applyRewrite134((Hop) hi);
		hi = _applyRewrite135((Hop) hi);
		hi = _applyRewrite136((Hop) hi);
		hi = _applyRewrite137((Hop) hi);
		hi = _applyRewrite138((Hop) hi);
		hi = _applyRewrite139((Hop) hi);
		hi = _applyRewrite140((Hop) hi);
		hi = _applyRewrite141((Hop) hi);
		hi = _applyRewrite142((Hop) hi);
		hi = _applyRewrite143((Hop) hi);
		hi = _applyRewrite144((Hop) hi);
		hi = _applyRewrite145((Hop) hi);
		hi = _applyRewrite146((Hop) hi);
		hi = _applyRewrite147((Hop) hi);
		hi = _applyRewrite148((Hop) hi);
		hi = _applyRewrite149((Hop) hi);
		hi = _applyRewrite150((Hop) hi);
		hi = _applyRewrite151((Hop) hi);
		hi = _applyRewrite152((Hop) hi);
		hi = _applyRewrite153((Hop) hi);
		hi = _applyRewrite154((Hop) hi);
		hi = _applyRewrite155((Hop) hi);
		hi = _applyRewrite156((Hop) hi);
		hi = _applyRewrite157((Hop) hi);
		hi = _applyRewrite158((Hop) hi);
		hi = _applyRewrite159((Hop) hi);
		hi = _applyRewrite160((Hop) hi);
		hi = _applyRewrite161((Hop) hi);
		hi = _applyRewrite162((Hop) hi);
		hi = _applyRewrite163((Hop) hi);
		hi = _applyRewrite164((Hop) hi);
		hi = _applyRewrite165((Hop) hi);
		hi = _applyRewrite166((Hop) hi);
		hi = _applyRewrite167((Hop) hi);
		hi = _applyRewrite168((Hop) hi);
		hi = _applyRewrite169((Hop) hi);
		hi = _applyRewrite170((Hop) hi);
		hi = _applyRewrite171((Hop) hi);
		hi = _applyRewrite172((Hop) hi);
		hi = _applyRewrite173((Hop) hi);
		hi = _applyRewrite174((Hop) hi);
		hi = _applyRewrite175((Hop) hi);
		hi = _applyRewrite176((Hop) hi);
		hi = _applyRewrite177((Hop) hi);
		hi = _applyRewrite178((Hop) hi);
		hi = _applyRewrite179((Hop) hi);
		hi = _applyRewrite180((Hop) hi);
		hi = _applyRewrite181((Hop) hi);
		hi = _applyRewrite182((Hop) hi);
		hi = _applyRewrite183((Hop) hi);
		hi = _applyRewrite184((Hop) hi);
		hi = _applyRewrite185((Hop) hi);
		hi = _applyRewrite186((Hop) hi);
		hi = _applyRewrite187((Hop) hi);
		hi = _applyRewrite188((Hop) hi);
		hi = _applyRewrite189((Hop) hi);
		hi = _applyRewrite190((Hop) hi);
		hi = _applyRewrite191((Hop) hi);
		hi = _applyRewrite192((Hop) hi);
		hi = _applyRewrite193((Hop) hi);
		hi = _applyRewrite194((Hop) hi);
		hi = _applyRewrite195((Hop) hi);
		hi = _applyRewrite196((Hop) hi);
		hi = _applyRewrite197((Hop) hi);
		hi = _applyRewrite198((Hop) hi);
		hi = _applyRewrite199((Hop) hi);
		hi = _applyRewrite200((Hop) hi);
		hi = _applyRewrite201((Hop) hi);
		hi = _applyRewrite202((Hop) hi);
		hi = _applyRewrite203((Hop) hi);
		hi = _applyRewrite204((Hop) hi);
		hi = _applyRewrite205((Hop) hi);
		hi = _applyRewrite206((Hop) hi);
		hi = _applyRewrite207((Hop) hi);
		hi = _applyRewrite208((Hop) hi);
		hi = _applyRewrite209((Hop) hi);
		hi = _applyRewrite210((Hop) hi);
		hi = _applyRewrite211((Hop) hi);
		hi = _applyRewrite212((Hop) hi);
		hi = _applyRewrite213((Hop) hi);
		hi = _applyRewrite214((Hop) hi);
		hi = _applyRewrite215((Hop) hi);
		hi = _applyRewrite216((Hop) hi);
		hi = _applyRewrite217((Hop) hi);
		hi = _applyRewrite218((Hop) hi);
		hi = _applyRewrite219((Hop) hi);
		hi = _applyRewrite220((Hop) hi);
		hi = _applyRewrite221((Hop) hi);
		hi = _applyRewrite222((Hop) hi);
		hi = _applyRewrite223((Hop) hi);
		hi = _applyRewrite224((Hop) hi);
		hi = _applyRewrite225((Hop) hi);
		hi = _applyRewrite226((Hop) hi);
		hi = _applyRewrite227((Hop) hi);
		hi = _applyRewrite228((Hop) hi);
		hi = _applyRewrite229((Hop) hi);
		hi = _applyRewrite230((Hop) hi);
		hi = _applyRewrite231((Hop) hi);
		hi = _applyRewrite232((Hop) hi);
		hi = _applyRewrite233((Hop) hi);
		hi = _applyRewrite234((Hop) hi);
		hi = _applyRewrite235((Hop) hi);
		hi = _applyRewrite236((Hop) hi);
		return hi;
	}

	// Implementation of the rule +(Z,0) => Z
	private static Hop _applyRewrite0(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
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
		System.out.println("Applying rewrite: +(Z,0) => Z");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_0);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return hi_0;
	}

	// Implementation of the rule +(0.0,Z) => Z
	private static Hop _applyRewrite1(Hop hi) {
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

		if ( hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(0.0,Z) => Z");

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, hi_1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return hi_1;
	}

	// Implementation of the rule +(%*%(is_LT_infinite,flip_pos),%*%(A,flip_pos)) => %*%(+(A,is_LT_infinite),flip_pos)
	private static Hop _applyRewrite2(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !HopRewriteUtils.isMatrixMultiply(hi_1) )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_0_1 != hi_1_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(%*%(is_LT_infinite,flip_pos),%*%(A,flip_pos)) => %*%(+(A,is_LT_infinite),flip_pos)");
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

	// Implementation of the rule /(/(*(A,b),c),d) => *(A,/(/(b,c),d))
	private static Hop _applyRewrite4(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(/(*(A,b),c),d) => *(A,/(/(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(*(/(A,c),b),d) => *(A,/(/(b,c),d))
	private static Hop _applyRewrite5(Hop hi) {
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

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(/(A,c),b),d) => *(A,/(/(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(*(b,/(A,c)),d) => *(A,/(/(b,c),d))
	private static Hop _applyRewrite6(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.DIV || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(b,/(A,c)),d) => *(A,/(/(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule *(/(/(A,c),d),b) => *(A,/(/(b,c),d))
	private static Hop _applyRewrite7(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(/(A,c),d),b) => *(A,/(/(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(/(*(lr,A),a),b) => *(/(*(lr,b),a),A)
	private static Hop _applyRewrite8(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(*(lr,A),a),b) => *(/(*(lr,b),a),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(/(*(A,lr),a),b) => *(/(*(lr,b),a),A)
	private static Hop _applyRewrite9(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(*(A,lr),a),b) => *(/(*(lr,b),a),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(lr,/(*(a,A),b)) => *(/(*(lr,a),b),A)
	private static Hop _applyRewrite10(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(lr,/(*(a,A),b)) => *(/(*(lr,a),b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(lr,/(*(A,a),b)) => *(/(*(lr,a),b),A)
	private static Hop _applyRewrite11(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(lr,/(*(A,a),b)) => *(/(*(lr,a),b),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule /(*(/(a,D),b),c) => /(*(a,/(b,c)),D)
	private static Hop _applyRewrite12(Hop hi) {
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

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(/(a,D),b),c) => /(*(a,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(*(a,/(b,D)),c) => /(*(a,/(b,c)),D)
	private static Hop _applyRewrite13(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.DIV || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(a,/(b,D)),c) => /(*(a,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule *(/(/(a,D),c),b) => /(*(a,/(b,c)),D)
	private static Hop _applyRewrite14(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(/(a,D),c),b) => /(*(a,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(a,/(/(b,D),c)) => /(*(a,/(b,c)),D)
	private static Hop _applyRewrite15(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(/(b,D),c)) => /(*(a,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0_0, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(/(A,parsertemp91781),N) => *(/(N,parsertemp91781),A)
	private static Hop _applyRewrite16(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(A,parsertemp91781),N) => *(/(N,parsertemp91781),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule *(N,/(A,parsertemp91781)) => *(/(N,parsertemp91781),A)
	private static Hop _applyRewrite17(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(N,/(A,parsertemp91781)) => *(/(N,parsertemp91781),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule /(*(N,A),parsertemp91781) => *(/(N,parsertemp91781),A)
	private static Hop _applyRewrite18(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(N,A),parsertemp91781) => *(/(N,parsertemp91781),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule /(*(A,N),parsertemp91781) => *(/(N,parsertemp91781),A)
	private static Hop _applyRewrite19(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(A,N),parsertemp91781) => *(/(N,parsertemp91781),A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule t(-(a,t(A))) => -(a,A)
	private static Hop _applyRewrite20(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(-(a,t(A))) => -(a,A)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule t(+(t(A),reg_covar)) => +(A,reg_covar)
	private static Hop _applyRewrite21(Hop hi) {
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

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(t(A),reg_covar)) => +(A,reg_covar)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule t(+(reg_covar,t(A))) => +(A,reg_covar)
	private static Hop _applyRewrite22(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(reg_covar,t(A))) => +(A,reg_covar)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule +(-(a,-(D,b)),c) => -(+(a,+(b,c)),D)
	private static Hop _applyRewrite23(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(a,-(D,b)),c) => -(+(a,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(a,-(b,-(D,c))) => -(+(a,+(b,c)),D)
	private static Hop _applyRewrite24(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(b,-(D,c))) => -(+(a,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_1_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(c,-(b,+(d,A))) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite25(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(b,+(d,A))) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(c,-(b,+(A,d))) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite26(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(b,+(A,d))) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(-(c,-(b,A)),d) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite27(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(c,-(b,A)),d) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(c,-(d,-(b,A))) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite28(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(d,-(b,A))) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(-(+(c,A),b),d) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite29(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(+(c,A),b),d) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(-(+(A,c),b),d) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite30(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(+(A,c),b),d) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(c,-(+(d,A),b)) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite31(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(+(d,A),b)) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(c,-(+(A,d),b)) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite32(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(+(A,d),b)) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(c,-(-(b,A),d)) => -(A,-(-(b,c),d))
	private static Hop _applyRewrite33(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(-(b,A),d)) => -(A,-(-(b,c),d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(+(-(a,D),c),b) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite34(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(-(a,D),c),b) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(+(a,-(c,D)),b) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite35(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(a,-(c,D)),b) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(a,+(-(D,c),b)) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite36(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(-(D,c),b)) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(a,+(b,-(D,c))) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite37(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(b,-(D,c))) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(a,-(+(b,D),c)) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite38(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(+(b,D),c)) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(a,-(+(D,b),c)) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite39(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(+(D,b),c)) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0_1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(-(-(a,D),b),c) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite40(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(-(a,D),b),c) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(a,-(-(c,D),b)) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite41(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(-(c,D),b)) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(-(a,+(b,D)),c) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite42(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(a,+(b,D)),c) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(-(a,+(D,b)),c) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite43(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(a,+(D,b)),c) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(a,-(c,+(b,D))) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite44(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(c,+(b,D))) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(a,-(c,+(D,b))) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite45(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(c,+(D,b))) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1_1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(-(a,-(D,c)),b) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite46(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,-(D,c)),b) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(a,-(b,-(c,D))) => -(-(a,-(b,c)),D)
	private static Hop _applyRewrite47(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(b,-(c,D))) => -(-(a,-(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(+(-(A,b),c),d) => -(A,-(b,-(c,d)))
	private static Hop _applyRewrite48(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(-(A,b),c),d) => -(A,-(b,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(+(c,-(A,b)),d) => -(A,-(b,-(c,d)))
	private static Hop _applyRewrite49(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(c,-(A,b)),d) => -(A,-(b,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(c,+(-(b,A),d)) => -(A,-(b,-(c,d)))
	private static Hop _applyRewrite50(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,+(-(b,A),d)) => -(A,-(b,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(c,+(b,-(d,A))) => -(A,-(b,-(c,d)))
	private static Hop _applyRewrite51(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,+(b,-(d,A))) => -(A,-(b,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(-(+(c,A),b),d) => -(A,-(b,-(c,d)))
	private static Hop _applyRewrite52(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(+(c,A),b),d) => -(A,-(b,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(c,-(-(A,b),d)) => -(A,-(b,-(c,d)))
	private static Hop _applyRewrite53(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(-(A,b),d)) => -(A,-(b,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(-(c,-(b,A)),d) => -(A,-(b,-(c,d)))
	private static Hop _applyRewrite54(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(c,-(b,A)),d) => -(A,-(b,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(c,-(b,-(A,d))) => -(A,-(b,-(c,d)))
	private static Hop _applyRewrite55(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(b,-(A,d))) => -(A,-(b,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(-(a,+(b,D)),c) => -(-(-(a,b),c),D)
	private static Hop _applyRewrite56(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,+(b,D)),c) => -(-(-(a,b),c),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(a,+(D,b)),c) => -(-(-(a,b),c),D)
	private static Hop _applyRewrite57(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,+(D,b)),c) => -(-(-(a,b),c),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule /(/(a,C),b) => /(/(a,b),C)
	private static Hop _applyRewrite58(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(/(a,C),b) => /(/(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule /(t(*(a,C)),b) => *(/(a,b),t(C))
	private static Hop _applyRewrite59(Hop hi) {
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

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(t(*(a,C)),b) => *(/(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(t(*(C,a)),b) => *(/(a,b),t(C))
	private static Hop _applyRewrite60(Hop hi) {
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

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(t(*(C,a)),b) => *(/(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(t(/(C,b)),a) => *(/(a,b),t(C))
	private static Hop _applyRewrite61(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(/(C,b)),a) => *(/(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0_1, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(a,t(/(C,b))) => *(/(a,b),t(C))
	private static Hop _applyRewrite62(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,t(/(C,b))) => *(/(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule /(t(/(a,C)),b) => /(/(a,b),t(C))
	private static Hop _applyRewrite63(Hop hi) {
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

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(t(/(a,C)),b) => /(/(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.DIV);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule %*%(colSums(A),/(C,b)) => %*%(/(colSums(A),b),C)
	private static Hop _applyRewrite64(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0 = (AggUnaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.AggOp.SUM || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi_0.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(colSums(A),/(C,b)) => %*%(/(colSums(A),b),C)");
		AggUnaryOp v1 = HopRewriteUtils.createAggUnaryOp(hi_0_0, Types.AggOp.SUM, Types.Direction.Col);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(v2, hi_1_0);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule t(*(t(vb3),beta2)) => *(beta2,vb3)
	private static Hop _applyRewrite65(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(*(t(vb3),beta2)) => *(beta2,vb3)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule t(*(beta2,t(vb3))) => *(beta2,vb3)
	private static Hop _applyRewrite66(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(*(beta2,t(vb3))) => *(beta2,vb3)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule /(/(*(b,A),D),c) => /(*(A,/(b,c)),D)
	private static Hop _applyRewrite67(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(/(*(b,A),D),c) => /(*(A,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(/(*(A,b),D),c) => /(*(A,/(b,c)),D)
	private static Hop _applyRewrite68(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(/(*(A,b),D),c) => /(*(A,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(*(/(b,D),A),c) => /(*(A,/(b,c)),D)
	private static Hop _applyRewrite69(Hop hi) {
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

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(/(b,D),A),c) => /(*(A,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(*(A,/(b,D)),c) => /(*(A,/(b,c)),D)
	private static Hop _applyRewrite70(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.DIV || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(A,/(b,D)),c) => /(*(A,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule *(/(/(A,c),D),b) => /(*(A,/(b,c)),D)
	private static Hop _applyRewrite71(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(/(A,c),D),b) => /(*(A,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(/(b,D),/(A,c)) => /(*(A,/(b,c)),D)
	private static Hop _applyRewrite72(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(b,D),/(A,c)) => /(*(A,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule *(/(A,c),/(b,D)) => /(*(A,/(b,c)),D)
	private static Hop _applyRewrite73(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(A,c),/(b,D)) => /(*(A,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule *(b,/(/(A,c),D)) => /(*(A,/(b,c)),D)
	private static Hop _applyRewrite74(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(b,/(/(A,c),D)) => /(*(A,/(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule /(/(/(a,C),D),b) => /(/(/(a,b),C),D)
	private static Hop _applyRewrite75(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(/(/(a,C),D),b) => /(/(/(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule t(/(a,t(parsertemp46794))) => /(a,parsertemp46794)
	private static Hop _applyRewrite76(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(a,t(parsertemp46794))) => /(a,parsertemp46794)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v1;
	}

	// Implementation of the rule t(/(t(A),a)) => /(A,a)
	private static Hop _applyRewrite77(Hop hi) {
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

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(t(A),a)) => /(A,a)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v1;
	}

	// Implementation of the rule *(/(a,C),b) => /(*(a,b),C)
	private static Hop _applyRewrite78(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(a,C),b) => /(*(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule *(a,/(b,C)) => /(*(a,b),C)
	private static Hop _applyRewrite79(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(b,C)) => /(*(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule *(t(*(b,A)),c) => *(t(A),*(b,c))
	private static Hop _applyRewrite80(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(*(b,A)),c) => *(t(A),*(b,c))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(t(*(A,b)),c) => *(t(A),*(b,c))
	private static Hop _applyRewrite81(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(*(A,b)),c) => *(t(A),*(b,c))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(b,t(*(c,A))) => *(t(A),*(b,c))
	private static Hop _applyRewrite82(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(b,t(*(c,A))) => *(t(A),*(b,c))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_1_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(b,t(*(A,c))) => *(t(A),*(b,c))
	private static Hop _applyRewrite83(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(b,t(*(A,c))) => *(t(A),*(b,c))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_1_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(0,-(parsertemp138264,R)) => -(R,parsertemp138264)
	private static Hop _applyRewrite84(Hop hi) {
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

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(0,-(parsertemp138264,R)) => -(R,parsertemp138264)");
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

	// Implementation of the rule -(-(A,b),c) => -(A,+(b,c))
	private static Hop _applyRewrite85(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,b),c) => -(A,+(b,c))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule -(-(a,C),b) => -(-(a,b),C)
	private static Hop _applyRewrite86(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,C),b) => -(-(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule -(a,+(b,C)) => -(-(a,b),C)
	private static Hop _applyRewrite87(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(b,C)) => -(-(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(a,+(C,b)) => -(-(a,b),C)
	private static Hop _applyRewrite88(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(C,b)) => -(-(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(a,-(C,b)) => -(+(a,b),C)
	private static Hop _applyRewrite89(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(C,b)) => -(+(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(a,C),b) => -(+(a,b),C)
	private static Hop _applyRewrite90(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(a,C),b) => -(+(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule +(a,-(b,C)) => -(+(a,b),C)
	private static Hop _applyRewrite91(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(b,C)) => -(+(a,b),C)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(int927,-(a,A)) => +(A,-(int927,a))
	private static Hop _applyRewrite92(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(int927,-(a,A)) => +(A,-(int927,a))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule +(-(A,a),int927) => +(A,-(int927,a))
	private static Hop _applyRewrite93(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,a),int927) => +(A,-(int927,a))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule +(int927,-(A,a)) => +(A,-(int927,a))
	private static Hop _applyRewrite94(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(int927,-(A,a)) => +(A,-(int927,a))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v2;
	}

	// Implementation of the rule -(+(int927,A),a) => +(A,-(int927,a))
	private static Hop _applyRewrite95(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(int927,A),a) => +(A,-(int927,a))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule -(+(A,int927),a) => +(A,-(int927,a))
	private static Hop _applyRewrite96(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,int927),a) => +(A,-(int927,a))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return v2;
	}

	// Implementation of the rule *(t(/(a,C)),b) => /(*(a,b),t(C))
	private static Hop _applyRewrite97(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(/(a,C)),b) => /(*(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(a,t(/(b,C))) => /(*(a,b),t(C))
	private static Hop _applyRewrite98(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,t(/(b,C))) => /(*(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(/(*(b,A),D),c) => *(A,/(*(b,c),D))
	private static Hop _applyRewrite99(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(*(b,A),D),c) => *(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(/(*(A,b),D),c) => *(A,/(*(b,c),D))
	private static Hop _applyRewrite100(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(*(A,b),D),c) => *(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(b,/(*(c,A),D)) => *(A,/(*(b,c),D))
	private static Hop _applyRewrite101(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(b,/(*(c,A),D)) => *(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule *(b,/(*(A,c),D)) => *(A,/(*(b,c),D))
	private static Hop _applyRewrite102(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(b,/(*(A,c),D)) => *(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_0, v2, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule t(/(*(parsertemp205616,t(H)),t(A))) => /(*(H,t(parsertemp205616)),A)
	private static Hop _applyRewrite103(Hop hi) {
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

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0_1 = (ReorgOp) hi_0_0_1;

		if ( c_hi_0_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1_0 = hi_0_0_1.getInput(0);

		if ( hi_0_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(*(parsertemp205616,t(H)),t(A))) => /(*(H,t(parsertemp205616)),A)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1_0, v1, Types.OpOp2.MULT);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0_1);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule *(/(/(a,C),D),b) => /(/(*(a,b),C),D)
	private static Hop _applyRewrite104(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
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

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(/(a,C),D),b) => /(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule *(/(a,C),/(b,D)) => /(/(*(a,b),C),D)
	private static Hop _applyRewrite105(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(/(a,C),/(b,D)) => /(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule *(a,/(/(b,C),D)) => /(/(*(a,b),C),D)
	private static Hop _applyRewrite106(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.DIV || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(a,/(/(b,C),D)) => /(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_0_1, Types.OpOp2.DIV);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule t(-(A,t(B))) => -(t(A),B)
	private static Hop _applyRewrite107(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(-(A,t(B))) => -(t(A),B)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule -(t(A),t(tmp)) => t(-(A,tmp))
	private static Hop _applyRewrite108(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(A),t(tmp)) => t(-(A,tmp))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
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

	// Implementation of the rule +(t(A),t(b4)) => t(+(A,b4))
	private static Hop _applyRewrite109(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(A),t(b4)) => t(+(A,b4))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.PLUS);
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

	// Implementation of the rule t(+(t(A),B)) => +(A,t(B))
	private static Hop _applyRewrite110(Hop hi) {
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

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(t(A),B)) => +(A,t(B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(+(A,t(B))) => +(B,t(A))
	private static Hop _applyRewrite111(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(+(A,t(B))) => +(B,t(A))");
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

	// Implementation of the rule t(-(t(A),parsertemp236854)) => -(A,t(parsertemp236854))
	private static Hop _applyRewrite112(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(-(t(A),parsertemp236854)) => -(A,t(parsertemp236854))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule -(t(-(A,b)),c) => -(t(A),+(b,c))
	private static Hop _applyRewrite113(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(-(A,b)),c) => -(t(A),+(b,c))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(t(+(a,C)),b) => +(-(a,b),t(C))
	private static Hop _applyRewrite114(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(+(a,C)),b) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(t(+(C,a)),b) => +(-(a,b),t(C))
	private static Hop _applyRewrite115(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(+(C,a)),b) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(t(-(C,b)),a) => +(-(a,b),t(C))
	private static Hop _applyRewrite116(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(-(C,b)),a) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(a,t(-(C,b))) => +(-(a,b),t(C))
	private static Hop _applyRewrite117(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,t(-(C,b))) => +(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(t(-(a,C)),b) => -(+(a,b),t(C))
	private static Hop _applyRewrite118(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(-(a,C)),b) => -(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(a,t(-(b,C))) => -(+(a,b),t(C))
	private static Hop _applyRewrite119(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,t(-(b,C))) => -(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(t(+(a,C)),b) => +(+(a,b),t(C))
	private static Hop _applyRewrite120(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(+(a,C)),b) => +(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(t(+(C,a)),b) => +(+(a,b),t(C))
	private static Hop _applyRewrite121(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(t(+(C,a)),b) => +(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(a,t(+(b,C))) => +(+(a,b),t(C))
	private static Hop _applyRewrite122(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,t(+(b,C))) => +(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(a,t(+(C,b))) => +(+(a,b),t(C))
	private static Hop _applyRewrite123(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,t(+(C,b))) => +(+(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_1_0_0);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.PLUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(a,+(-(D,b),C)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite124(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(-(D,b),C)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(a,+(D,-(C,b))) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite125(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,+(D,-(C,b))) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(-(-(a,D),C),b) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite126(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(-(a,D),C),b) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_0_0_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(-(a,D),-(b,C)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite127(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(a,D),-(b,C)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(a,-(-(b,C),D)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite128(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(a,-(-(b,C),D)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(-(a,C),-(D,b)) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite129(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,C),-(D,b)) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(a,-(D,-(b,C))) => -(+(a,b),+(C,D))
	private static Hop _applyRewrite130(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(a,-(D,-(b,C))) => -(+(a,b),+(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(+(-(A,c),B),d) => -(+(A,B),+(c,d))
	private static Hop _applyRewrite131(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(-(A,c),B),d) => -(+(A,B),+(c,d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(+(A,-(B,c)),d) => -(+(A,B),+(c,d))
	private static Hop _applyRewrite132(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,-(B,c)),d) => -(+(A,B),+(c,d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(-(B,c),-(A,d)) => -(+(A,B),+(c,d))
	private static Hop _applyRewrite133(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(B,c),-(A,d)) => -(+(A,B),+(c,d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(B,-(c,A)),d) => -(+(A,B),+(c,d))
	private static Hop _applyRewrite134(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(B,-(c,A)),d) => -(+(A,B),+(c,d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(B,c),-(d,A)) => -(+(A,B),+(c,d))
	private static Hop _applyRewrite135(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(B,c),-(d,A)) => -(+(A,B),+(c,d))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(t(-(a,C)),b) => -(-(a,b),t(C))
	private static Hop _applyRewrite136(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(t(-(a,C)),b) => -(-(a,b),t(C))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		ReorgOp v2 = HopRewriteUtils.createTranspose(hi_0_0_1);
		BinaryOp v3 = HopRewriteUtils.createBinary(v1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(+(b,A),-(D,c)) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite137(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(b,A),-(D,c)) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(A,b),-(D,c)) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite138(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,b),-(D,c)) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(b,-(D,+(c,A))) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite139(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(D,+(c,A))) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(b,-(D,+(A,c))) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite140(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(D,+(A,c))) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(-(A,-(D,b)),c) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite141(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,-(D,b)),c) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(b,-(A,-(D,c))) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite142(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(A,-(D,c))) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(-(+(b,A),D),c) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite143(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(+(b,A),D),c) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(-(+(A,b),D),c) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite144(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(+(A,b),D),c) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(b,-(+(c,A),D)) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite145(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(+(c,A),D)) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(b,-(+(A,c),D)) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite146(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(b,-(+(A,c),D)) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(b,-(-(D,c),A)) => -(+(A,+(b,c)),D)
	private static Hop _applyRewrite147(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(b,-(-(D,c),A)) => -(+(A,+(b,c)),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(+(c,D),-(b,A)) => -(A,-(-(b,c),D))
	private static Hop _applyRewrite148(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(c,D),-(b,A)) => -(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(A,c),-(b,D)) => -(A,-(-(b,c),D))
	private static Hop _applyRewrite149(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,c),-(b,D)) => -(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(-(A,-(b,D)),c) => -(A,-(-(b,c),D))
	private static Hop _applyRewrite150(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,-(b,D)),c) => -(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(c,-(A,-(b,D))) => -(A,-(-(b,c),D))
	private static Hop _applyRewrite151(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(A,-(b,D))) => -(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(c,-(-(b,A),D)) => -(A,-(-(b,c),D))
	private static Hop _applyRewrite152(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(-(b,A),D)) => -(A,-(-(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0_0, hi_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(+(-(c,B),A),d) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite153(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(-(c,B),A),d) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(+(A,-(c,B)),d) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite154(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,-(c,B)),d) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(c,+(-(d,A),B)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite155(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,+(-(d,A),B)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(c,+(B,-(d,A))) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite156(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,+(B,-(d,A))) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(-(+(c,A),B),d) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite157(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(+(c,A),B),d) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(-(+(A,c),B),d) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite158(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.PLUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(+(A,c),B),d) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(c,-(+(d,B),A)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite159(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.FP64 && hi_1_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(+(d,B),A)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(c,-(+(B,d),A)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite160(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.PLUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(+(B,d),A)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule -(+(c,A),+(d,B)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite161(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(c,A),+(d,B)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(c,A),+(B,d)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite162(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(c,A),+(B,d)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(A,c),+(d,B)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite163(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,c),+(d,B)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(+(A,c),+(B,d)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite164(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(+(A,c),+(B,d)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(-(-(A,d),B),c) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite165(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(-(A,d),B),c) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule +(-(c,B),-(A,d)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite166(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(c,B),-(A,d)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(-(A,d),-(c,B)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite167(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,d),-(c,B)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule +(c,-(-(A,d),B)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite168(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MINUS || c_hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.SCALAR || (hi_1_0_1.getValueType() != Types.ValueType.FP64 && hi_1_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(-(A,d),B)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return v3;
	}

	// Implementation of the rule +(-(A,+(d,B)),c) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite169(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,+(d,B)),c) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(-(A,+(B,d)),c) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite170(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(-(A,+(B,d)),c) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule +(c,-(A,+(d,B))) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite171(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_1_0.getValueType() != Types.ValueType.FP64 && hi_1_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(A,+(d,B))) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule +(c,-(A,+(B,d))) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite172(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.PLUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(c,-(A,+(B,d))) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(-(A,-(B,c)),d) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite173(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.MINUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,-(B,c)),d) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(c,B),-(d,A)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite174(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(c,B),-(d,A)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(A,d),-(B,c)) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite175(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,d),-(B,c)) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(c,-(B,-(A,d))) => -(A,-(B,-(c,d)))
	private static Hop _applyRewrite176(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || (hi_0.getValueType() != Types.ValueType.FP64 && hi_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( !(hi_1_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_1 = (BinaryOp) hi_1_1;

		if ( c_hi_1_1.getOp() != Types.OpOp2.MINUS || c_hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_0 = hi_1_1.getInput(0);

		if ( hi_1_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1_1 = hi_1_1.getInput(1);

		if ( hi_1_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(c,-(B,-(A,d))) => -(A,-(B,-(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_1_1_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_1);

		return v3;
	}

	// Implementation of the rule -(-(a,C),+(b,D)) => -(-(-(a,b),C),D)
	private static Hop _applyRewrite177(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,C),+(b,D)) => -(-(-(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(a,C),+(D,b)) => -(-(-(a,b),C),D)
	private static Hop _applyRewrite178(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(a,C),+(D,b)) => -(-(-(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_1_0, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(-(a,C),D),b) => -(-(-(a,b),C),D)
	private static Hop _applyRewrite179(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(-(a,C),D),b) => -(-(-(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_1, Types.OpOp2.MINUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1, Types.OpOp2.MINUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(v2, hi_0_1, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule -(-(A,+(c,B)),d) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite180(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.SCALAR || (hi_0_1_0.getValueType() != Types.ValueType.FP64 && hi_0_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,+(c,B)),d) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(A,+(B,c)),d) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite181(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_1 = (BinaryOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.OpOp2.PLUS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_1 = hi_0_1.getInput(1);

		if ( hi_0_1_1.getDataType() != Types.DataType.SCALAR || (hi_0_1_1.getValueType() != Types.ValueType.FP64 && hi_0_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,+(B,c)),d) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule -(-(A,c),+(d,B)) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite182(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,c),+(d,B)) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(A,c),+(B,d)) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite183(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.PLUS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(A,c),+(B,d)) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule -(-(-(A,c),B),d) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite184(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_0_1.getValueType() != Types.ValueType.FP64 && hi_0_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || (hi_1.getValueType() != Types.ValueType.FP64 && hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: -(-(-(A,c),B),d) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MINUS);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(scale_lambda,1000) => *(scale_lambda,0.001)
	private static Hop _applyRewrite185(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 1000 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(scale_lambda,1000) => *(scale_lambda,0.001)");
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

	// Implementation of the rule /(A,100000) => *(A,1.0E-5)
	private static Hop _applyRewrite186(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 100000 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,100000) => *(A,1.0E-5)");
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

	// Implementation of the rule /(A,100) => *(0.01,A)
	private static Hop _applyRewrite187(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 100 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,100) => *(0.01,A)");
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

	// Implementation of the rule /(parsertemp6002,0.5) => *(2.0,parsertemp6002)
	private static Hop _applyRewrite188(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.5 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(parsertemp6002,0.5) => *(2.0,parsertemp6002)");
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

	// Implementation of the rule /(parsertemp14437,10000) => *(parsertemp14437,1.0E-4)
	private static Hop _applyRewrite189(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.INT64 && l_hi_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1.getLongValue() != 10000 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(parsertemp14437,10000) => *(parsertemp14437,1.0E-4)");
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

	// Implementation of the rule /(A,2.0) => *(0.5,A)
	private static Hop _applyRewrite190(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR || (l_hi_1.getValueType() != Types.ValueType.FP64 && l_hi_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		if ( l_hi_1.getDoubleValue() != 2.0 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(A,2.0) => *(0.5,A)");
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

	// Implementation of the rule /(A,2) => *(0.5,A)
	private static Hop _applyRewrite191(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
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
		System.out.println("Applying rewrite: /(A,2) => *(0.5,A)");
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

	// Implementation of the rule rowSums(-(a,t(B))) => t(colSums(-(a,B)))
	private static Hop _applyRewrite192(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(-(a,t(B))) => t(colSums(-(a,B)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule rowSums(-(t(A),b)) => t(colSums(-(A,b)))
	private static Hop _applyRewrite193(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(-(t(A),b)) => t(colSums(-(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MINUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(-(a,t(B))) => t(rowSums(-(a,B)))
	private static Hop _applyRewrite194(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(-(a,t(B))) => t(rowSums(-(a,B)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.MINUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule colSums(-(t(A),b)) => t(rowSums(-(A,b)))
	private static Hop _applyRewrite195(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(-(t(A),b)) => t(rowSums(-(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MINUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule rowSums(+(t(A),b)) => t(colSums(+(A,b)))
	private static Hop _applyRewrite196(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(+(t(A),b)) => t(colSums(+(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule rowSums(+(b,t(A))) => t(colSums(+(A,b)))
	private static Hop _applyRewrite197(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(+(b,t(A))) => t(colSums(+(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.PLUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule colSums(+(t(A),b)) => t(rowSums(+(A,b)))
	private static Hop _applyRewrite198(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(+(t(A),b)) => t(rowSums(+(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.PLUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(+(b,t(A))) => t(rowSums(+(A,b)))
	private static Hop _applyRewrite199(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.PLUS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(+(b,t(A))) => t(rowSums(+(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.PLUS);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule *(t(neighbors),t(border)) => t(*(neighbors,border))
	private static Hop _applyRewrite200(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: *(t(neighbors),t(border)) => t(*(neighbors,border))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
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

	// Implementation of the rule t(*(t(G),c)) => *(G,t(c))
	private static Hop _applyRewrite201(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(*(t(G),c)) => *(G,t(c))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule t(*(c,t(G))) => *(G,t(c))
	private static Hop _applyRewrite202(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(*(c,t(G))) => *(G,t(c))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1_0, v1, Types.OpOp2.MULT);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v2;
	}

	// Implementation of the rule rowSums(*(t(A),b)) => t(colSums(*(A,b)))
	private static Hop _applyRewrite203(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(*(t(A),b)) => t(colSums(*(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule rowSums(*(b,t(A))) => t(colSums(*(A,b)))
	private static Hop _applyRewrite204(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(*(b,t(A))) => t(colSums(*(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.MULT);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule colSums(*(t(A),b)) => t(rowSums(*(A,b)))
	private static Hop _applyRewrite205(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(*(t(A),b)) => t(rowSums(*(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.MULT);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(*(b,t(A))) => t(rowSums(*(A,b)))
	private static Hop _applyRewrite206(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(*(b,t(A))) => t(rowSums(*(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1_0, hi_0_0, Types.OpOp2.MULT);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule t(/(t(A),weight)) => /(A,t(weight))
	private static Hop _applyRewrite207(Hop hi) {
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

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(t(A),weight)) => /(A,t(weight))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_0_0, v1, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v2);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v2;
	}

	// Implementation of the rule /(t(A),t(B)) => t(/(A,B))
	private static Hop _applyRewrite208(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(t(A),t(B)) => t(/(A,B))");
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

	// Implementation of the rule t(/(A,t(B))) => /(t(A),B)
	private static Hop _applyRewrite209(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(A,t(B))) => /(t(A),B)");
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

	// Implementation of the rule rowSums(/(a,t(B))) => t(colSums(/(a,B)))
	private static Hop _applyRewrite210(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(/(a,t(B))) => t(colSums(/(a,B)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule rowSums(/(t(A),b)) => t(colSums(/(A,b)))
	private static Hop _applyRewrite211(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: rowSums(/(t(A),b)) => t(colSums(/(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Col);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule colSums(/(a,t(B))) => t(rowSums(/(a,B)))
	private static Hop _applyRewrite212(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(/(a,t(B))) => t(rowSums(/(a,B)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_0_1_0, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_1);

		return v3;
	}

	// Implementation of the rule colSums(/(t(A),b)) => t(rowSums(/(A,b)))
	private static Hop _applyRewrite213(Hop hi) {
		if ( !(hi instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi = (AggUnaryOp) hi;

		if ( c_hi.getOp() != Types.AggOp.SUM || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi.getDirection() == Types.Direction.Col) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: colSums(/(t(A),b)) => t(rowSums(/(A,b)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_0, hi_0_1, Types.OpOp2.DIV);
		AggUnaryOp v2 = HopRewriteUtils.createAggUnaryOp(v1, Types.AggOp.SUM, Types.Direction.Row);
		ReorgOp v3 = HopRewriteUtils.createTranspose(v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return v3;
	}

	// Implementation of the rule /(*(weight,t(A)),t(B)) => *(t(/(A,B)),weight)
	private static Hop _applyRewrite214(Hop hi) {
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

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(*(weight,t(A)),t(B)) => *(t(/(A,B)),weight)");
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

	// Implementation of the rule t(/(*(A,t(weight)),B)) => *(t(/(A,B)),weight)
	private static Hop _applyRewrite215(Hop hi) {
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

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0_1 = (ReorgOp) hi_0_0_1;

		if ( c_hi_0_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1_0 = hi_0_0_1.getInput(0);

		if ( hi_0_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(*(A,t(weight)),B)) => *(t(/(A,B)),weight)");
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

	// Implementation of the rule %*%(*(c,A),/(B,d)) => %*%(A,*(B,/(c,d)))
	private static Hop _applyRewrite216(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(c,A),/(B,d)) => %*%(A,*(B,/(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MULT);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_1, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(A,c),/(B,d)) => %*%(A,*(B,/(c,d)))
	private static Hop _applyRewrite217(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(A,c),/(B,d)) => %*%(A,*(B,/(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MULT);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(/(A,d),*(c,B)) => %*%(A,*(B,/(c,d)))
	private static Hop _applyRewrite218(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(/(A,d),*(c,B)) => %*%(A,*(B,/(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.MULT);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(/(A,d),*(B,c)) => %*%(A,*(B,/(c,d)))
	private static Hop _applyRewrite219(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(/(A,d),*(B,c)) => %*%(A,*(B,/(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MULT);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(/(a,C),/(D,b)) => %*%(/(/(a,b),C),D)
	private static Hop _applyRewrite220(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(/(a,C),/(D,b)) => %*%(/(/(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(v2, hi_1_0);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(/(A,c),/(b,D)) => %*%(A,/(/(b,c),D))
	private static Hop _applyRewrite221(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(/(A,c),/(b,D)) => %*%(A,/(/(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0, hi_0_1, Types.OpOp2.DIV);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule t(/(%*%(t(V),W),t(parsertemp63810))) => /(%*%(t(W),V),parsertemp63810)
	private static Hop _applyRewrite222(Hop hi) {
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

		if ( hi_0_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(/(%*%(t(V),W),t(parsertemp63810))) => /(%*%(t(W),V),parsertemp63810)");
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

	// Implementation of the rule %*%(*(c,A),*(d,B)) => %*%(A,*(B,*(c,d)))
	private static Hop _applyRewrite223(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(c,A),*(d,B)) => %*%(A,*(B,*(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.MULT);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_1, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(c,A),*(B,d)) => %*%(A,*(B,*(c,d)))
	private static Hop _applyRewrite224(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(c,A),*(B,d)) => %*%(A,*(B,*(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MULT);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_1, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(A,c),*(d,B)) => %*%(A,*(B,*(c,d)))
	private static Hop _applyRewrite225(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(A,c),*(d,B)) => %*%(A,*(B,*(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_1, v1, Types.OpOp2.MULT);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(A,c),*(B,d)) => %*%(A,*(B,*(c,d)))
	private static Hop _applyRewrite226(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(A,c),*(B,d)) => %*%(A,*(B,*(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1_0, v1, Types.OpOp2.MULT);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(/(a,C),*(b,D)) => %*%(/(*(a,b),C),D)
	private static Hop _applyRewrite227(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(/(a,C),*(b,D)) => %*%(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(v2, hi_1_1);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(/(a,C),*(D,b)) => %*%(/(*(a,b),C),D)
	private static Hop _applyRewrite228(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MULT || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.FP64 && hi_1_1.getValueType() != Types.ValueType.FP32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(/(a,C),*(D,b)) => %*%(/(*(a,b),C),D)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_1, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(v2, hi_1_0);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(b,A),/(c,D)) => %*%(A,/(*(b,c),D))
	private static Hop _applyRewrite229(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || (hi_0_0.getValueType() != Types.ValueType.FP64 && hi_0_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(b,A),/(c,D)) => %*%(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_1, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(*(A,b),/(c,D)) => %*%(A,/(*(b,c),D))
	private static Hop _applyRewrite230(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MULT || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || (hi_0_1.getValueType() != Types.ValueType.FP64 && hi_0_1.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || (hi_1_0.getValueType() != Types.ValueType.FP64 && hi_1_0.getValueType() != Types.ValueType.FP32) )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(*(A,b),/(c,D)) => %*%(A,/(*(b,c),D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_1, hi_1_0, Types.OpOp2.MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_1_1, Types.OpOp2.DIV);
		AggBinaryOp v3 = HopRewriteUtils.createMatrixMultiply(hi_0_0, v2);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v3);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return v3;
	}

	// Implementation of the rule %*%(t(y),t(parsertemp11966)) => t(%*%(parsertemp11966,y))
	private static Hop _applyRewrite231(Hop hi) {
		if ( !HopRewriteUtils.isMatrixMultiply(hi) )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0 = (ReorgOp) hi_0;

		if ( c_hi_0.getOp() != Types.ReOrgOp.TRANS || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_1 = (ReorgOp) hi_1;

		if ( c_hi_1.getOp() != Types.ReOrgOp.TRANS || c_hi_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: %*%(t(y),t(parsertemp11966)) => t(%*%(parsertemp11966,y))");
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

	// Implementation of the rule t(%*%(t(A),p)) => %*%(t(p),A)
	private static Hop _applyRewrite232(Hop hi) {
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

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(%*%(t(A),p)) => %*%(t(p),A)");
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

	// Implementation of the rule t(%*%(A,t(X))) => %*%(X,t(A))
	private static Hop _applyRewrite233(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( !(hi_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_1 = (ReorgOp) hi_0_1;

		if ( c_hi_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_1_0 = hi_0_1.getInput(0);

		if ( hi_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(%*%(A,t(X))) => %*%(X,t(A))");
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

	// Implementation of the rule t(rowSums(/(A,t(B)))) => colSums(/(t(A),B))
	private static Hop _applyRewrite234(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof AggUnaryOp) )
			return hi;

		AggUnaryOp c_hi_0 = (AggUnaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.AggOp.SUM || c_hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		if ( !(c_hi_0.getDirection() == Types.Direction.Row) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.DIV || c_hi_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( !(hi_0_0_1 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0_1 = (ReorgOp) hi_0_0_1;

		if ( c_hi_0_0_1.getOp() != Types.ReOrgOp.TRANS || c_hi_0_0_1.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0_0_1_0 = hi_0_0_1.getInput(0);

		if ( hi_0_0_1_0.getDataType() != Types.DataType.MATRIX )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: t(rowSums(/(A,t(B)))) => colSums(/(t(A),B))");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_0_0);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_0_1_0, Types.OpOp2.DIV);
		AggUnaryOp v3 = HopRewriteUtils.createAggUnaryOp(v2, Types.AggOp.SUM, Types.Direction.Col);

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

	// Implementation of the rule /(parsertemp264984,+(-(sample_block_size,1),1)) => /(parsertemp264984,sample_block_size)
	private static Hop _applyRewrite235(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
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

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || (hi_1_0_0.getValueType() != Types.ValueType.INT64 && hi_1_0_0.getValueType() != Types.ValueType.INT32) )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( !(hi_1_0_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0_1 = (LiteralOp) hi_1_0_1;

		if ( l_hi_1_0_1.getDataType() != Types.DataType.SCALAR || (l_hi_1_0_1.getValueType() != Types.ValueType.INT64 && l_hi_1_0_1.getValueType() != Types.ValueType.INT32) )
			return hi;

		if ( l_hi_1_0_1.getLongValue() != 1 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_0_1 != hi_1_1 )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: /(parsertemp264984,+(-(sample_block_size,1),1)) => /(parsertemp264984,sample_block_size)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0_0, Types.OpOp2.DIV);

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, v1);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0_1);

		return v1;
	}

	// Implementation of the rule +(A,-(0,a)) => -(A,a)
	private static Hop _applyRewrite236(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || c_hi.getDataType() != Types.DataType.MATRIX )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX )
			return hi;

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

		if ( hi_1_1.getDataType() != Types.DataType.SCALAR || (hi_1_1.getValueType() != Types.ValueType.INT64 && hi_1_1.getValueType() != Types.ValueType.INT32) )
			return hi;


		// Now, we start building the new Hop
		System.out.println("Applying rewrite: +(A,-(0,a)) => -(A,a)");
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
