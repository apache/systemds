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

	// state flags
	private boolean _cacheInProgress = true; // caching in progress, in the first pass.
	private int _replayPosition = 0; // slider position in the stream

	public ResettableStream(LocalTaskQueue<IndexedMatrixValue> source) {
		this._source = source;
		this._cache = new ArrayList<>();
	}

	@Override
	public synchronized IndexedMatrixValue dequeueTask()
					throws InterruptedException {
		// Implement dequeTask logic
		return null;
	}

	/**
	 * Resets the stream to beginning to read the stream from start.
	 * This can only be called once the stream is fully consumed once.
	 */
	public void reset() {
		// Implement reset logic
	}

	@Override
	public synchronized void closeInput() {

		_source.closeInput();
	}
}
