package org.tugraz.sysds.runtime.compress.utils;

import java.util.ArrayList;
import java.util.Iterator;

import org.tugraz.sysds.runtime.compress.ColGroup;
import org.tugraz.sysds.runtime.matrix.data.IJV;

public class ColumnGroupIterator implements Iterator<IJV> {
	// iterator configuration
	private final int _rl;
	private final int _ru;
	private final int _cgu;
	private final boolean _inclZeros;

	// iterator state
	private int _posColGroup = -1;
	private Iterator<IJV> _iterColGroup = null;
	private boolean _noNext = false;
	private ArrayList<ColGroup> _colGroups;

	public ColumnGroupIterator(int rl, int ru, int cgl, int cgu, boolean inclZeros, ArrayList<ColGroup> colGroups) {
		_rl = rl;
		_ru = ru;
		_cgu = cgu;
		_inclZeros = inclZeros;
		_posColGroup = cgl - 1;
		_colGroups = colGroups;
		getNextIterator();
	}

	@Override
	public boolean hasNext() {
		return !_noNext;
	}

	@Override
	public IJV next() {
		if(_noNext)
			throw new RuntimeException("No more entries.");
		IJV ret = _iterColGroup.next();
		if(!_iterColGroup.hasNext())
			getNextIterator();
		return ret;
	}

	private void getNextIterator() {
		while(_posColGroup + 1 < _cgu) {
			_posColGroup++;
			_iterColGroup = _colGroups.get(_posColGroup).getIterator(_rl, _ru, _inclZeros, false);
			if(_iterColGroup.hasNext())
				return;
		}
		_noNext = true;
	}
}
