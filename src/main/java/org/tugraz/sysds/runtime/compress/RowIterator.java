package org.tugraz.sysds.runtime.compress;

import java.util.ArrayList;
import java.util.Iterator;

import org.tugraz.sysds.runtime.compress.ColGroup.ColGroupRowIterator;

abstract class RowIterator<T> implements Iterator<T> {
	// iterator configuration
	protected final int _rl;
	protected final int _ru;

	private final ArrayList<ColGroup> _colGroups;

	// iterator state
	protected ColGroupRowIterator[] _iters = null;
	protected int _rpos;

	public RowIterator(int rl, int ru, ArrayList<ColGroup> colGroups) {
		_rl = rl;
		_ru = ru;
		_colGroups = colGroups;

		// initialize array of column group iterators
		_iters = new ColGroupRowIterator[_colGroups.size()];
		for(int i = 0; i < _colGroups.size(); i++)
			_iters[i] = _colGroups.get(i).getRowIterator(_rl, _ru);

		// get initial row
		_rpos = rl;
	}

	@Override
	public boolean hasNext() {
		return(_rpos < _ru);
	}
}
