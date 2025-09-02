package org.apache.sysds.runtime.controlprogram.parfor;

import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;
import org.apache.sysds.runtime.DMLRuntimeException;
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

	/**
	 * Dequeues a task. If it is the first, it reads from the disk and stores in the cache.
	 * For subsequent passes it reads from the memory.
	 *
	 * @return The next matrix value in the stream, or NO_MORE_TASKS
	 * @throws InterruptedException
	 */
	@Override
	public synchronized IndexedMatrixValue dequeueTask()
					throws InterruptedException {
		if (_cacheInProgress) {
			// First pass: Read value from the source and cache it, and return.
			IndexedMatrixValue task = _source.dequeueTask();
			if (task != NO_MORE_TASKS) {
				_cache.add(new IndexedMatrixValue(task));
			} else {
				_cacheInProgress = false; // caching is complete
				_source.closeInput(); // close source stream
			}
			notifyAll(); // Notify all the waiting consumers waiting for cache to fill with this stream
			return task;
		} else {
			// Replay pass: read directly from in-memory cache
			if (_replayPosition < _cache.size()) {
				// Return a copy to ensure consumer won't modify the cache
				return new IndexedMatrixValue(_cache.get(_replayPosition++));
			} else {
				return (IndexedMatrixValue) NO_MORE_TASKS;
			}
		}
	}

	/**
	 * Resets the stream to beginning to read the stream from start.
	 * This can only be called once the stream is fully consumed once.
	 */
	public synchronized void reset() throws InterruptedException {
		if (_cacheInProgress) {
			System.out.println("Attempted to reset a stream that's not been fully cached yet.");
			wait();
//			throw new DMLRuntimeException("Attempted to reset a stream that's not been fully cached yet.");
		}
		_replayPosition = 0;
	}

	@Override
	public synchronized void closeInput() {

		_source.closeInput();
	}
}
