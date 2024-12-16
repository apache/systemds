package org.apache.sysds.hops.rewriter;

import java.util.ArrayList;
import java.util.function.Function;

import org.apache.sysds.utils.Statistics;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewriter.dml.DMLExecutor;

public class GeneratedRewriteClass implements Function {

	@Override
	public Object apply( Object _hi ) {
		if ( _hi == null )
			return null;

		Hop hi = (Hop) _hi;

		if ( hi.getDataType() == Types.DataType.MATRIX ) {
			if ( hi instanceof BinaryOp ) {
				if ( (( BinaryOp ) hi ).getOp() == Types.OpOp2.MULT ) {
					if ( hi.getInput().size() == 2 ) {
						Hop hi_0 = hi.getInput(0);
						Hop hi_1 = hi.getInput(1);
						if ( hi_0.getDataType() == Types.DataType.SCALAR ) {
							hi = _applyRewrite0(hi); // *(0.0,A) => const(A,0.0)
						} else if ( hi_0.getDataType() == Types.DataType.MATRIX ) {
							hi = _applyRewrite2(hi); // *(/(1.0,B),a) => /(a,B)
							hi = _applyRewrite3(hi); // *(/(1.0,B),A) => /(A,B)
							hi = _applyRewrite7(hi); // *(A,0.0) => const(A,0.0)
							hi = _applyRewrite9(hi); // *(cast.MATRIX(a),b) => cast.MATRIX(*(a,b))
							hi = _applyRewrite16(hi); // *(A,/(1.0,B)) => /(A,B)
						}
					}
				} else if ( (( BinaryOp ) hi ).getOp() == Types.OpOp2.MINUS ) {
					hi = _applyRewrite1(hi); // -(-(-(A,c),B),d) => -(A,+(B,+(c,d)))
					hi = _applyRewrite5(hi); // -(a,cast.MATRIX(b)) => cast.MATRIX(-(a,b))
					hi = _applyRewrite13(hi); // -(0.0,-(B,A)) => -(A,B)
					hi = _applyRewrite14(hi); // -(A,/(*(b,C),D)) => -*(A,b,/(C,D))
				} else if ( (( BinaryOp ) hi ).getOp() == Types.OpOp2.DIV ) {
					hi = _applyRewrite8(hi); // /(a,cast.MATRIX(b)) => cast.MATRIX(/(a,b))
				} else if ( (( BinaryOp ) hi ).getOp() == Types.OpOp2.PLUS ) {
					hi = _applyRewrite10(hi); // +(A,0.0) => A
					hi = _applyRewrite11(hi); // +(-(*(C,b),d),A) => -(+*(A,b,C),d)
					hi = _applyRewrite12(hi); // +(-(*(D,c),B),A) => -(A,-*(B,c,D))
					hi = _applyRewrite17(hi); // +(-(0.0,B),A) => -(A,B)
				}
			} else if ( hi instanceof ReorgOp ) {
				hi = _applyRewrite6(hi); // t(%*%(t(B),A)) => %*%(t(A),B)
			}
		} else if ( hi.getDataType() == Types.DataType.SCALAR ) {
			hi = _applyRewrite4(hi); // *(0.0,a) => 0.0
			hi = _applyRewrite15(hi); // /(0.0,a) => 0.0
		}
		return hi;
	}

	// Implementation of the rule *(0.0,A) => const(A,0.0)
	private static Hop _applyRewrite0(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("*(0.0,A) => const(A,0.0)");
		DataGenOp v1 = ((DataGenOp) HopRewriteUtils.createDataGenOpFomDims(HopRewriteUtils.createUnary(hi_1, Types.OpOp1.NROW),HopRewriteUtils.createUnary(hi_1, Types.OpOp1.NCOL),0.0D));

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);

		return newRoot;
	}

	// Implementation of the rule -(-(-(A,c),B),d) => -(A,+(B,+(c,d)))
	private static Hop _applyRewrite1(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MINUS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("-(-(-(A,c),B),d) => -(A,+(B,+(c,d)))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0_1, hi_1, Types.OpOp2.PLUS);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_0_1, v1, Types.OpOp2.PLUS);
		BinaryOp v3 = HopRewriteUtils.createBinary(hi_0_0_0, v2, Types.OpOp2.MINUS);

		Hop newRoot = v3;
		if ( v3.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return newRoot;
	}

	// Implementation of the rule *(/(1.0,B),a) => /(a,B)
	private static Hop _applyRewrite2(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("*(/(1.0,B),a) => /(a,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.DIV);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return newRoot;
	}

	// Implementation of the rule *(/(1.0,B),A) => /(A,B)
	private static Hop _applyRewrite3(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.DIV || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("*(/(1.0,B),A) => /(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.DIV);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return newRoot;
	}

	// Implementation of the rule *(0.0,a) => 0.0
	private static Hop _applyRewrite4(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("*(0.0,a) => 0.0");

		Hop newRoot = hi_0;
		if ( hi_0.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return newRoot;
	}

	// Implementation of the rule -(a,cast.MATRIX(b)) => cast.MATRIX(-(a,b))
	private static Hop _applyRewrite5(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof UnaryOp) )
			return hi;

		UnaryOp c_hi_1 = (UnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp1.CAST_AS_MATRIX || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("-(a,cast.MATRIX(b)) => cast.MATRIX(-(a,b))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.MINUS);
		UnaryOp v2 = HopRewriteUtils.createUnary(v1, Types.OpOp1.CAST_AS_MATRIX);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return newRoot;
	}

	// Implementation of the rule t(%*%(t(B),A)) => %*%(t(A),B)
	private static Hop _applyRewrite6(Hop hi) {
		if ( !(hi instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi = (ReorgOp) hi;

		if ( c_hi.getOp() != Types.ReOrgOp.TRANS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !HopRewriteUtils.isMatrixMultiply(hi_0) )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof ReorgOp) )
			return hi;

		ReorgOp c_hi_0_0 = (ReorgOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.ReOrgOp.TRANS || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;


		if ( hi_0_0_0.getDim2() == -1 || hi_0_1.getDim2() == -1 || hi_0_0_0.getNnz() == -1 || hi_0_0_0.getDim1() == -1 || hi_0_1.getNnz() == -1 || hi_0_1.getDim1() == -1 )
			return hi;


		double costFrom = (hi_0_0_0.getNnz() + (Math.min((hi_0_0_0.getDim2() * hi_0_0_0.getDim1()), hi_0_1.getNnz()) * hi_0_0_0.getDim1() * 3.0) + (Math.min(((hi_0_0_0.getDim2() * hi_0_0_0.getDim1()) * (1.0 / hi_0_0_0.getDim2())), 1.0) * Math.min((hi_0_1.getNnz() * (1.0 / hi_0_1.getDim2())), 1.0) * hi_0_0_0.getDim2() * hi_0_1.getDim2()) + 30030.0);
		double costTo = (hi_0_1.getNnz() + (Math.min((hi_0_1.getDim2() * hi_0_1.getDim1()), hi_0_0_0.getNnz()) * hi_0_0_0.getDim1() * 3.0) + 20020.0);

		if ( costFrom <= costTo )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("t(%*%(t(B),A)) => %*%(t(A),B)");
		ReorgOp v1 = HopRewriteUtils.createTranspose(hi_0_1);
		AggBinaryOp v2 = HopRewriteUtils.createMatrixMultiply(v1, hi_0_0_0);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return newRoot;
	}

	// Implementation of the rule *(A,0.0) => const(A,0.0)
	private static Hop _applyRewrite7(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("*(A,0.0) => const(A,0.0)");
		DataGenOp v1 = ((DataGenOp) HopRewriteUtils.createDataGenOpFomDims(HopRewriteUtils.createUnary(hi_0, Types.OpOp1.NROW),HopRewriteUtils.createUnary(hi_0, Types.OpOp1.NCOL),0.0D));

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);

		return newRoot;
	}

	// Implementation of the rule /(a,cast.MATRIX(b)) => cast.MATRIX(/(a,b))
	private static Hop _applyRewrite8(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.SCALAR || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof UnaryOp) )
			return hi;

		UnaryOp c_hi_1 = (UnaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp1.CAST_AS_MATRIX || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.SCALAR || !hi_1_0.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("/(a,cast.MATRIX(b)) => cast.MATRIX(/(a,b))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_0, Types.OpOp2.DIV);
		UnaryOp v2 = HopRewriteUtils.createUnary(v1, Types.OpOp1.CAST_AS_MATRIX);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return newRoot;
	}

	// Implementation of the rule *(cast.MATRIX(a),b) => cast.MATRIX(*(a,b))
	private static Hop _applyRewrite9(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof UnaryOp) )
			return hi;

		UnaryOp c_hi_0 = (UnaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp1.CAST_AS_MATRIX || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( hi_0_0.getDataType() != Types.DataType.SCALAR || !hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("*(cast.MATRIX(a),b) => cast.MATRIX(*(a,b))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0_0, hi_1, Types.OpOp2.MULT);
		UnaryOp v2 = HopRewriteUtils.createUnary(v1, Types.OpOp1.CAST_AS_MATRIX);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);

		return newRoot;
	}

	// Implementation of the rule +(A,0.0) => A
	private static Hop _applyRewrite10(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( !(hi_1 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1 = (LiteralOp) hi_1;

		if ( l_hi_1.getDataType() != Types.DataType.SCALAR|| !l_hi_1.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1.getDoubleValue() != 0.0 )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("+(A,0.0) => A");

		Hop newRoot = hi_0;
		if ( hi_0.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return newRoot;
	}

	// Implementation of the rule +(-(*(C,b),d),A) => -(+*(A,b,C),d)
	private static Hop _applyRewrite11(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("+(-(*(C,b),d),A) => -(+*(A,b,C),d)");
		TernaryOp v1 = HopRewriteUtils.createTernary(hi_1, hi_0_0_1, hi_0_0_0,Types.OpOp3.PLUS_MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(v1, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return newRoot;
	}

	// Implementation of the rule +(-(*(D,c),B),A) => -(A,-*(B,c,D))
	private static Hop _applyRewrite12(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if (hi_0_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0_0 = (BinaryOp) hi_0_0;

		if ( c_hi_0_0.getOp() != Types.OpOp2.MULT || !c_hi_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_0 = hi_0_0.getInput(0);

		if ( hi_0_0_0.getDataType() != Types.DataType.MATRIX || !hi_0_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0_1 = hi_0_0.getInput(1);

		if ( hi_0_0_1.getDataType() != Types.DataType.SCALAR || !hi_0_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("+(-(*(D,c),B),A) => -(A,-*(B,c,D))");
		TernaryOp v1 = HopRewriteUtils.createTernary(hi_0_1, hi_0_0_1, hi_0_0_0,Types.OpOp3.MINUS_MULT);
		BinaryOp v2 = HopRewriteUtils.createBinary(hi_1, v1, Types.OpOp2.MINUS);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return newRoot;
	}

	// Implementation of the rule -(0.0,-(B,A)) => -(A,B)
	private static Hop _applyRewrite13(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.MINUS || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( hi_1_0.getDataType() != Types.DataType.MATRIX || !hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("-(0.0,-(B,A)) => -(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_1, hi_1_0, Types.OpOp2.MINUS);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return newRoot;
	}

	// Implementation of the rule -(A,/(*(b,C),D)) => -*(A,b,/(C,D))
	private static Hop _applyRewrite14(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MINUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if (hi_1_0.getParent().size() > 1)
			return hi;
		if ( !(hi_1_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1_0 = (BinaryOp) hi_1_0;

		if ( c_hi_1_0.getOp() != Types.OpOp2.MULT || !c_hi_1_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_0 = hi_1_0.getInput(0);

		if ( hi_1_0_0.getDataType() != Types.DataType.SCALAR || !hi_1_0_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0_1 = hi_1_0.getInput(1);

		if ( hi_1_0_1.getDataType() != Types.DataType.MATRIX || !hi_1_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("-(A,/(*(b,C),D)) => -*(A,b,/(C,D))");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1_0_1, hi_1_1, Types.OpOp2.DIV);
		TernaryOp v2 = HopRewriteUtils.createTernary(hi_0, hi_1_0_0, v1,Types.OpOp3.MINUS_MULT);

		Hop newRoot = v2;
		if ( v2.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return newRoot;
	}

	// Implementation of the rule /(0.0,a) => 0.0
	private static Hop _applyRewrite15(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.DIV || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( !(hi_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0 = (LiteralOp) hi_0;

		if ( l_hi_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.SCALAR || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("/(0.0,a) => 0.0");

		Hop newRoot = hi_0;
		if ( hi_0.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);

		return newRoot;
	}

	// Implementation of the rule *(A,/(1.0,B)) => /(A,B)
	private static Hop _applyRewrite16(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.MULT || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if ( hi_0.getDataType() != Types.DataType.MATRIX || !hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if (hi_1.getParent().size() > 1)
			return hi;
		if ( !(hi_1 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_1 = (BinaryOp) hi_1;

		if ( c_hi_1.getOp() != Types.OpOp2.DIV || !c_hi_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1_0 = hi_1.getInput(0);

		if ( !(hi_1_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_1_0 = (LiteralOp) hi_1_0;

		if ( l_hi_1_0.getDataType() != Types.DataType.SCALAR|| !l_hi_1_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_1_0.getDoubleValue() != 1.0 )
			return hi;

		Hop hi_1_1 = hi_1.getInput(1);

		if ( hi_1_1.getDataType() != Types.DataType.MATRIX || !hi_1_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("*(A,/(1.0,B)) => /(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_0, hi_1_1, Types.OpOp2.DIV);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_1);
		HopRewriteUtils.cleanupUnreferenced(hi_1_0);

		return newRoot;
	}

	// Implementation of the rule +(-(0.0,B),A) => -(A,B)
	private static Hop _applyRewrite17(Hop hi) {
		if ( !(hi instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi = (BinaryOp) hi;

		if ( c_hi.getOp() != Types.OpOp2.PLUS || !c_hi.getValueType().isNumeric() )
			return hi;

		Hop hi_0 = hi.getInput(0);

		if (hi_0.getParent().size() > 1)
			return hi;
		if ( !(hi_0 instanceof BinaryOp) )
			return hi;

		BinaryOp c_hi_0 = (BinaryOp) hi_0;

		if ( c_hi_0.getOp() != Types.OpOp2.MINUS || !c_hi_0.getValueType().isNumeric() )
			return hi;

		Hop hi_0_0 = hi_0.getInput(0);

		if ( !(hi_0_0 instanceof LiteralOp) )
			return hi;

		LiteralOp l_hi_0_0 = (LiteralOp) hi_0_0;

		if ( l_hi_0_0.getDataType() != Types.DataType.SCALAR|| !l_hi_0_0.getValueType().isNumeric() )
			return hi;

		if ( l_hi_0_0.getDoubleValue() != 0.0 )
			return hi;

		Hop hi_0_1 = hi_0.getInput(1);

		if ( hi_0_1.getDataType() != Types.DataType.MATRIX || !hi_0_1.getValueType().isNumeric() )
			return hi;

		Hop hi_1 = hi.getInput(1);

		if ( hi_1.getDataType() != Types.DataType.MATRIX || !hi_1.getValueType().isNumeric() )
			return hi;


		// Now, we start building the new Hop
		Statistics.applyGeneratedRewrite("+(-(0.0,B),A) => -(A,B)");
		BinaryOp v1 = HopRewriteUtils.createBinary(hi_1, hi_0_1, Types.OpOp2.MINUS);

		Hop newRoot = v1;
		if ( v1.getValueType() != hi.getValueType() ) {
			newRoot = castIfNecessary(newRoot, hi);
			if ( newRoot == null )
				return hi;
		}

		ArrayList<Hop> parents = new ArrayList<>(hi.getParent());

		for ( Hop p : parents )
			HopRewriteUtils.replaceChildReference(p, hi, newRoot);

		// Remove old unreferenced Hops
		HopRewriteUtils.cleanupUnreferenced(hi);
		HopRewriteUtils.cleanupUnreferenced(hi_0);
		HopRewriteUtils.cleanupUnreferenced(hi_0_0);

		return newRoot;
	}

	private static Hop castIfNecessary(Hop newRoot, Hop oldRoot) {
		Types.OpOp1 cast = null;
		switch ( oldRoot.getValueType().toExternalString() ) {
			case "DOUBLE":
				cast = Types.OpOp1.CAST_AS_DOUBLE;
				break;
			case "INT":
				cast = Types.OpOp1.CAST_AS_INT;
				break;
			case "BOOLEAN":
				cast = Types.OpOp1.CAST_AS_BOOLEAN;
				break;
			default:
				return null;
		}

		return new UnaryOp("tmp", oldRoot.getDataType(), oldRoot.getValueType(), cast, newRoot);
	}
}