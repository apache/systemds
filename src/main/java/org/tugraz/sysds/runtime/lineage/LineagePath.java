package org.tugraz.sysds.runtime.lineage;

import java.util.BitSet;

public class LineagePath {
	protected BitSet _lastBranch = null;
	protected int _branchCounter = 0;
	
	
	public void initLastBranch() {
		_lastBranch = new BitSet();
		_branchCounter = 0;
	}
	
	public void removeLastBranch() {
		_lastBranch = null;
		_branchCounter = 0;
	}
	
	public void clearLastBranch() {
		if (_lastBranch != null)
			_lastBranch.clear();
		_branchCounter = 0;
	}
	
	public void setBranchPredicateValue(boolean value) {
		if (_lastBranch != null) {
			_lastBranch.set(_branchCounter++, value);
		}
	}
	
	public Long getLastBranch() {
		if (_lastBranch != null)
			if (_lastBranch.toLongArray().length > 1)
				throw new RuntimeException("Length of _lastBranch.toLongArray() exceeds 1!");
			else
				return _lastBranch.toLongArray().length == 1 ? _lastBranch.toLongArray()[0] : 0L;
		return 0L;
	}
}
