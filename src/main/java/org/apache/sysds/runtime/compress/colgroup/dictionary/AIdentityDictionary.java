package org.apache.sysds.runtime.compress.colgroup.dictionary;

import org.apache.sysds.runtime.compress.DMLCompressionException;

public abstract class AIdentityDictionary extends ACachingMBDictionary {
	/** The number of rows or columns, rows can be +1 if withEmpty is set. */
	protected final int nRowCol;
	/** Specify if the Identity matrix should contain an empty row in the end. */
	protected final boolean withEmpty;

	/**
	 * Create an identity matrix dictionary. It behaves as if allocated a Sparse Matrix block but exploits that the
	 * structure is known to have certain properties.
	 * 
	 * @param nRowCol The number of rows and columns in this identity matrix.
	 */
	public AIdentityDictionary(int nRowCol) {
		if(nRowCol <= 0)
			throw new DMLCompressionException("Invalid Identity Dictionary");
		this.nRowCol = nRowCol;
		this.withEmpty = false;
	}

	public AIdentityDictionary(int nRowCol, boolean withEmpty) {
		if(nRowCol <= 0)
			throw new DMLCompressionException("Invalid Identity Dictionary");
		this.nRowCol = nRowCol;
		this.withEmpty = withEmpty;
	}

	public boolean withEmpty() {
		return withEmpty;
	}

	public static long getInMemorySize(int numberColumns) {
		return 4 + 4 + 8; // int + padding + softReference
	}

	@Override
	public final boolean containsValue(double pattern) {
		return pattern == 0.0 || pattern == 1.0;
	}

	@Override
	public double[] productAllRowsToDouble(int nCol) {
		return new double[nRowCol + (withEmpty ? 1 : 0)];
	}

	@Override
	public double[] productAllRowsToDoubleWithDefault(double[] defaultTuple) {
		double[] ret = new double[nRowCol + (withEmpty ? 1 : 0) + 1];
		ret[ret.length - 1] = 1;
		for(int i = 0; i < defaultTuple.length; i++)
			ret[ret.length - 1] *= defaultTuple[i];
		return ret;
	}
}
