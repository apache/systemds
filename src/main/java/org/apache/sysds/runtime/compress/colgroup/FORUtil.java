package org.apache.sysds.runtime.compress.colgroup;

import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class FORUtil {

	protected static final double refSum(double[] reference) {
		double ret = 0;
		for(double d : reference)
			ret += d;
		return ret;
	}

	protected static final double refSumSq(double[] reference) {
		double ret = 0;
		for(double d : reference)
			ret += d * d;
		return ret;
	}

	protected final static boolean allZero(double[] in) {
		for(double v : in)
			if(v != 0)
				return false;
		return true;
	}

	protected final static boolean containsInfOrNan(double pattern, double[] reference) {
		if(Double.isNaN(pattern)) {
			for(double d : reference)
				if(Double.isNaN(d))
					return true;
			return false;
		}
		else {
			for(double d : reference)
				if(Double.isInfinite(d))
					return true;
			return false;
		}
	}

	protected final static double[] createReference(int nCol, double val) {
		double[] reference = new double[nCol];
		for(int i = 0; i < nCol; i++)
			reference[i] = val;
		return reference;
	}

	protected final static double[] unaryOperator(UnaryOperator op, double[] reference) {
		final double[] newRef = new double[reference.length];
		for(int i = 0; i < reference.length; i++)
			newRef[i] = op.fn.execute(reference[i]);
		return newRef;
	}
}
