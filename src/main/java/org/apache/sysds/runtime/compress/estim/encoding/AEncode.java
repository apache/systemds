package org.apache.sysds.runtime.compress.estim.encoding;

public abstract class AEncode implements IEncode {

	@Override
	public boolean equals(Object e) {
		return e instanceof IEncode && this.equals((IEncode) e);
	}
}
