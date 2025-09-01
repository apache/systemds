package org.apache.sysds.runtime.controlprogram.parfor;

import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;

import java.util.ArrayList;

/**
 * A wrapper around LocalTaskQueue to consume the source stream and reset to
 * consume again for other operators.
 *
 */
public class ResettableStream extends LocalTaskQueue<IndexedMatrixValue> {

	// original live stream
	private final LocalTaskQueue<IndexedMatrixValue> _source;

	// in-memory cache to store stream for re-play
	private final ArrayList<IndexedMatrixValue> _cache;

	public ResettableStream(LocalTaskQueue<IndexedMatrixValue> source) {
		this._source = source;
		this._cache = new ArrayList<>();
	}



}
