package org.apache.sysml.parser;

/**
 * These are the builtin constants
 */
public enum BuiltinConstant {
	PI(Math.PI),
	INF(Double.POSITIVE_INFINITY),
	NaN(Double.NaN);

	private DoubleIdentifier _id;

	BuiltinConstant(double d) {
		this._id = new DoubleIdentifier(d);
	}

	public DoubleIdentifier get() {
		return this._id;
	}

	public static boolean contains(String name) {
		for (BuiltinConstant c : BuiltinConstant.values()) {
			if (c.name().equals(name)) {
				return true;
			}
		}
		return false;
	}
}
