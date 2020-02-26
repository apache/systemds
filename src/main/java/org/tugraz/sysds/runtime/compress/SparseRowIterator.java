package org.tugraz.sysds.runtime.compress;

import java.util.ArrayList;

import org.tugraz.sysds.runtime.data.SparseRow;
import org.tugraz.sysds.runtime.data.SparseRowVector;

class SparseRowIterator extends RowIterator<SparseRow> {
	private final SparseRowVector _ret;
	private final double[] _tmp;

	public SparseRowIterator(int rl, int ru, ArrayList<ColGroup> colGroups, int clen) {
		super(rl, ru, colGroups);
		_ret = new SparseRowVector(clen);
		_tmp = new double[clen];
	}

	@Override
	public SparseRow next() {
		// prepare meta data common across column groups
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int ix = _rpos % blksz;
		final boolean last = (_rpos + 1 == _ru);
		// copy group rows into consolidated dense vector
		// to avoid binary search+shifting or final sort
		for(int j = 0; j < _iters.length; j++)
			_iters[j].next(_tmp, _rpos, ix, last);
		// append non-zero values to consolidated sparse row
		_ret.setSize(0);
		for(int i = 0; i < _tmp.length; i++)
			_ret.append(i, _tmp[i]);
		// advance to next row and return buffer
		_rpos++;
		return _ret;
	}
}
