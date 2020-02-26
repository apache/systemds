package org.tugraz.sysds.runtime.compress;

import java.util.ArrayList;
import java.util.Arrays;

class DenseRowIterator extends RowIterator<double[]> {

	private final double[] _ret;

	public DenseRowIterator(int rl, int ru, ArrayList<ColGroup> colGroups, int clen) {
		super(rl, ru, colGroups);
		_ret = new double[clen];
	}

	@Override
	public double[] next() {
		// prepare meta data common across column groups
		final int blksz = BitmapEncoder.BITMAP_BLOCK_SZ;
		final int ix = _rpos % blksz;
		final boolean last = (_rpos + 1 == _ru);
		// copy group rows into consolidated row
		Arrays.fill(_ret, 0);
		for(int j = 0; j < _iters.length; j++)
			_iters[j].next(_ret, _rpos, ix, last);
		// advance to next row and return buffer
		_rpos++;
		return _ret;
	}
}