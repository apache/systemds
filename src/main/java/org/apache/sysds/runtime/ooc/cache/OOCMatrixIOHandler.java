/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.ooc.cache;

import org.apache.sysds.api.DMLScript;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.ooc.stats.OOCEventLog;
import org.apache.sysds.runtime.ooc.stream.SourceOOCStream;
import org.apache.sysds.runtime.util.FastBufferedDataInputStream;
import org.apache.sysds.runtime.util.FastBufferedDataOutputStream;
import org.apache.sysds.runtime.util.LocalFileUtils;
import org.apache.sysds.utils.Statistics;
import scala.Tuple2;
import scala.Tuple3;

import java.io.DataInput;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.channels.Channels;
import java.nio.channels.ClosedByInterruptException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.AtomicReference;

public class OOCMatrixIOHandler implements OOCIOHandler {
	private static final int WRITER_SIZE = 4;
	private static final int READER_SIZE = 10;
	private static final long OVERFLOW = 8192 * 1024;
	private static final long MAX_PARTITION_SIZE = 8192 * 8192;

	private final String _spillDir;
	private final ThreadPoolExecutor _writeExec;
	private final ThreadPoolExecutor _readExec;
	private final ThreadPoolExecutor _srcReadExec;
	private final ThreadPoolExecutor _deleteExec;
	private final ConcurrentHashMap<BlockKey, ReadTask> _pendingReads = new ConcurrentHashMap<>();
	private final AtomicLong _readSeq = new AtomicLong(0);

	// Spill related structures
	private final ConcurrentHashMap<String, SpillLocation> _spillLocations =  new ConcurrentHashMap<>();
	private final ConcurrentHashMap<Integer, PartitionFile> _partitions = new ConcurrentHashMap<>();
	private final ConcurrentHashMap<BlockKey, SourceBlockDescriptor> _sourceLocations = new ConcurrentHashMap<>();
	private final AtomicInteger _partitionCounter = new AtomicInteger(0);
	private final Object _spillLock = new Object();
	private final CloseableQueue<Tuple2<BlockEntry, CompletableFuture<Void>>>[] _q;
	private final AtomicLong _wCtr;
	private final AtomicBoolean _started;

	private final int _evictCallerId = OOCEventLog.registerCaller("write");
	private final int _readCallerId = OOCEventLog.registerCaller("read");
	private final int _srcReadCallerId = OOCEventLog.registerCaller("read_src");

	@SuppressWarnings("unchecked")
	public OOCMatrixIOHandler() {
		this._spillDir = LocalFileUtils.getUniqueWorkingDir("ooc_stream");
		_writeExec = new ThreadPoolExecutor(
			WRITER_SIZE,
			WRITER_SIZE,
			0L,
			TimeUnit.MILLISECONDS,
			new ArrayBlockingQueue<>(100000));
		_readExec = new ThreadPoolExecutor(
			READER_SIZE,
			READER_SIZE,
			0L,
			TimeUnit.MILLISECONDS,
			new PriorityBlockingQueue<>());
		_srcReadExec = new ThreadPoolExecutor(
			READER_SIZE,
			READER_SIZE,
			0L,
			TimeUnit.MILLISECONDS,
			new ArrayBlockingQueue<>(100000));
		_deleteExec = new ThreadPoolExecutor(
			1,
			1,
			0L,
			TimeUnit.MILLISECONDS,
			new ArrayBlockingQueue<>(100000));
		_q = new CloseableQueue[WRITER_SIZE];
		_wCtr = new AtomicLong(0);
		_started = new  AtomicBoolean(false);
	}

	private synchronized void start() {
		if (_started.compareAndSet(false, true)) {
			for (int i = 0; i < WRITER_SIZE; i++) {
				final int finalIdx = i;
				_q[i] = new CloseableQueue<>();
				_writeExec.submit(() -> evictTask(_q[finalIdx]));
			}
		}
	}

	@Override
	public void shutdown() {
		boolean started = _started.get();
		if (started) {
			try {
				for(int i = 0; i < WRITER_SIZE; i++) {
					_q[i].close();
				}
			}
			catch(InterruptedException ignored) {
			}
		}
		_writeExec.getQueue().clear();
		_writeExec.shutdownNow();
		_readExec.getQueue().clear();
		_readExec.shutdownNow();
		_srcReadExec.getQueue().clear();
		_srcReadExec.shutdownNow();
		_deleteExec.getQueue().clear();
		_deleteExec.shutdownNow();
		_spillLocations.clear();
		_partitions.clear();
		if (started)
			LocalFileUtils.deleteFileIfExists(_spillDir);
	}

	@Override
	public CompletableFuture<Void> scheduleEviction(BlockEntry block) {
		start();
		CompletableFuture<Void> future = new CompletableFuture<>();
		try {
			long q = _wCtr.getAndAdd(block.getSize()) / OVERFLOW;
			int i = (int)(q % WRITER_SIZE);
			_q[i].enqueueIfOpen(new Tuple2<>(block, future));
		}
		catch(InterruptedException ignored) {
		}

		return future;
	}

	@Override
	public CompletableFuture<BlockEntry> scheduleRead(final BlockEntry block) {
		final CompletableFuture<BlockEntry> future = new CompletableFuture<>();
		int pinnedPartitionId = pinPartitionForRead(block.getKey());
		try {
			ReadTask task = new ReadTask(block, future, _readSeq.getAndIncrement(), pinnedPartitionId);
			_pendingReads.put(block.getKey(), task);
			_readExec.execute(task);
		} catch (RejectedExecutionException e) {
			unpinPartitionForRead(pinnedPartitionId);
			_pendingReads.remove(block.getKey());
			future.completeExceptionally(e);
		}
		return future;
	}

	@Override
	public void prioritizeRead(BlockKey key, double priority) {
		if (priority == 0)
			return;
		ReadTask task = _pendingReads.get(key);
		if (task == null)
			return;
		if (_readExec.getQueue().remove(task)) {
			task.addPriority(priority);
			_readExec.getQueue().offer(task);
		}
	}

	@Override
	public CompletableFuture<Boolean> scheduleDeletion(BlockEntry block) {
		removeSpillLocation(block.getKey().toFileKey());
		_sourceLocations.remove(block.getKey());
		return CompletableFuture.completedFuture(true);
	}

	@Override
	public void registerSourceLocation(BlockKey key, SourceBlockDescriptor descriptor) {
		_sourceLocations.put(key, descriptor);
	}

	@Override
	public CompletableFuture<SourceReadResult> scheduleSourceRead(SourceReadRequest request) {
		return submitSourceRead(request, null, request.maxBytesInFlight);
	}

	@Override
	public CompletableFuture<SourceReadResult> continueSourceRead(SourceReadContinuation continuation, long maxBytesInFlight) {
		if (!(continuation instanceof SourceReadState state)) {
			CompletableFuture<SourceReadResult> failed = new CompletableFuture<>();
			failed.completeExceptionally(new DMLRuntimeException("Unsupported continuation type: " + continuation));
			return failed;
		}
		return submitSourceRead(state.request, state, maxBytesInFlight);
	}

	private CompletableFuture<SourceReadResult> submitSourceRead(SourceReadRequest request, SourceReadState state,
		long maxBytesInFlight) {
		if(request.format != Types.FileFormat.BINARY)
			return CompletableFuture.failedFuture(
				new DMLRuntimeException("Unsupported format for source read: " + request.format));
		return readBinarySourceParallel(request, state, maxBytesInFlight);
	}

	private CompletableFuture<SourceReadResult> readBinarySourceParallel(SourceReadRequest request,
		SourceReadState state, long maxBytesInFlight) {
		final long byteLimit = maxBytesInFlight > 0 ? maxBytesInFlight : Long.MAX_VALUE;
		final AtomicLong bytesRead = new AtomicLong(0);
		final AtomicBoolean stop = new AtomicBoolean(false);
		final AtomicBoolean budgetHit = new AtomicBoolean(false);
		final AtomicReference<Throwable> error = new AtomicReference<>();
		final Object budgetLock = new Object();
		final CompletableFuture<SourceReadResult> result = new CompletableFuture<>();
		final ConcurrentLinkedDeque<SourceBlockDescriptor> descriptors = new ConcurrentLinkedDeque<>();

		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(request.path);

		Path[] files;
		AtomicLongArray filePositions;
		AtomicIntegerArray completed;

		try {
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			MatrixReader.checkValidInputFile(fs, path);

			if(state == null) {
				List<Path> seqFiles = new ArrayList<>(Arrays.asList(IOUtilFunctions.getSequenceFilePaths(fs, path)));
				files = seqFiles.toArray(Path[]::new);
				filePositions = new AtomicLongArray(files.length);
				completed = new AtomicIntegerArray(files.length);
			}
			else {
				files = state.paths;
				filePositions = state.filePositions;
				completed = state.completed;
			}
		}
		catch(IOException e) {
			throw new DMLRuntimeException(e);
		}

		int activeTasks = 0;
		for(int i = 0; i < files.length; i++)
			if(completed.get(i) == 0)
				activeTasks++;

		final AtomicInteger remaining = new AtomicInteger(activeTasks);
		boolean anyTask = activeTasks > 0;

		for(int i = 0; i < files.length; i++) {
			if(completed.get(i) == 1)
				continue;
			final int fileIdx = i;
			try {
				_srcReadExec.submit(() -> {
					try {
						readSequenceFile(job, files[fileIdx], request, fileIdx, filePositions, completed, stop,
							budgetHit, bytesRead, byteLimit, budgetLock, descriptors);
					}
					catch(Throwable t) {
						error.compareAndSet(null, t);
						stop.set(true);
					}
					finally {
						if(remaining.decrementAndGet() == 0)
							completeResult(result, bytesRead, budgetHit, error, request, files, filePositions,
								completed, descriptors);
					}
				});
			}
			catch(RejectedExecutionException e) {
				error.compareAndSet(null, e);
				stop.set(true);
				if(remaining.decrementAndGet() == 0)
					completeResult(result, bytesRead, budgetHit, error, request, files, filePositions, completed,
						descriptors);
				break;
			}
		}

		if(!anyTask) {
			try {
				closeTarget(request.target, true);
				result.complete(new SourceReadResult(bytesRead.get(), true, null, List.of()));
			}
			catch(DMLRuntimeException e) {
				result.completeExceptionally(e);
			}
		}

		return result;
	}

	private void completeResult(CompletableFuture<SourceReadResult> future, AtomicLong bytesRead, AtomicBoolean budgetHit,
		AtomicReference<Throwable> error, SourceReadRequest request, Path[] files, AtomicLongArray filePositions,
		AtomicIntegerArray completed, ConcurrentLinkedDeque<SourceBlockDescriptor> descriptors) {
		Throwable err = error.get();
		if (err != null) {
			future.completeExceptionally(err instanceof Exception ? err : new Exception(err));
			return;
		}

		try {
			if (budgetHit.get()) {
				if(!request.keepOpenOnLimit) {
					closeTarget(request.target, false);
				}
				SourceReadContinuation cont = new SourceReadState(request, files, filePositions, completed);
				future.complete(new SourceReadResult(bytesRead.get(), false, cont, new ArrayList<>(descriptors)));

				return;
			}

			closeTarget(request.target, true);
			future.complete(new SourceReadResult(bytesRead.get(), true, null, new ArrayList<>(descriptors)));
		}
		catch(DMLRuntimeException e) {
			future.completeExceptionally(e);
		}
	}

	private void readSequenceFile(JobConf job, Path path, SourceReadRequest request, int fileIdx,
		AtomicLongArray filePositions, AtomicIntegerArray completed, AtomicBoolean stop, AtomicBoolean budgetHit,
		AtomicLong bytesRead, long byteLimit, Object budgetLock, ConcurrentLinkedDeque<SourceBlockDescriptor> descriptors)
		throws IOException {
		MatrixIndexes key = new MatrixIndexes();
		MatrixBlock value = new MatrixBlock();

		try(SequenceFile.Reader reader = new SequenceFile.Reader(job, SequenceFile.Reader.file(path))) {
			long pos = filePositions.get(fileIdx);
			if (pos > 0)
				reader.seek(pos);

			long ioStart = DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
			while(!stop.get()) {
				long recordStart = reader.getPosition();
				if (!reader.next(key, value))
					break;
				long recordEnd = reader.getPosition();
				long blockSize = value.getExactSerializedSize();
				boolean shouldBreak = false;

				synchronized(budgetLock) {
					if (stop.get())
						shouldBreak = true;
					else if (bytesRead.get() + blockSize > byteLimit) {
						stop.set(true);
						budgetHit.set(true);
						shouldBreak = true;
					}
					bytesRead.addAndGet(blockSize);
				}

				MatrixIndexes outIdx = new MatrixIndexes(key);
				MatrixBlock outBlk = new MatrixBlock(value);
				IndexedMatrixValue imv = new IndexedMatrixValue(outIdx, outBlk);
				SourceBlockDescriptor descriptor = new SourceBlockDescriptor(path.toString(), request.format, outIdx,
					recordStart, (int)(recordEnd - recordStart), blockSize);

				if (request.target instanceof SourceOOCStream src)
					src.enqueue(imv, descriptor);
				else
					request.target.enqueue(imv);

				descriptors.add(descriptor);
				filePositions.set(fileIdx, reader.getPosition());

				if (DMLScript.OOC_LOG_EVENTS) {
					long currTime = System.nanoTime();
					OOCEventLog.onDiskReadEvent(_srcReadCallerId, ioStart, currTime, blockSize);
					ioStart = currTime;
				}

				if (shouldBreak)
					break; // Note that we knowingly go over limit, which could result in READER_SIZE*8MB overshoot
			}

			if (!stop.get())
				completed.set(fileIdx, 1);
		}
	}

	private void closeTarget(org.apache.sysds.runtime.instructions.ooc.OOCStream<IndexedMatrixValue> target, boolean close) {
		if(close) {
			try {
				target.closeInput();
			}
			catch(Exception ex) {
				throw ex instanceof DMLRuntimeException ? (DMLRuntimeException) ex : new DMLRuntimeException(ex);
			}
		}
	}


	private void loadFromDisk(BlockEntry block) {
		String key = block.getKey().toFileKey();

		SourceBlockDescriptor src = _sourceLocations.get(block.getKey());
		if (src != null) {
			long ioStart = DMLScript.OOC_STATISTICS ? System.nanoTime() : 0;
			loadFromSource(block, src);
			if (DMLScript.OOC_STATISTICS) {
				Statistics.incrementOOCLoadFromDisk();
				Statistics.accumulateOOCLoadFromDiskTime(System.nanoTime() - ioStart);
			}
			return;
		}

		long ioDuration = 0;
		// 1. find the blocks address (spill location)
		SpillLocation sloc = _spillLocations.get(key);
		if (sloc == null)
			throw new DMLRuntimeException("Failed to load spill location for: " + key);

		PartitionFile partFile = _partitions.get(sloc.partitionId);
		if (partFile == null)
			throw new DMLRuntimeException("Failed to load partition for: " + sloc.partitionId);

		String filename = partFile.filePath;

		// Create an empty object to read data into.
		MatrixIndexes ix = new  MatrixIndexes();
		MatrixBlock mb = new  MatrixBlock();

		try (RandomAccessFile raf = new RandomAccessFile(filename, "r")) {
			raf.seek(sloc.offset);

			DataInput dis = new FastBufferedDataInputStream(Channels.newInputStream(raf.getChannel()));
			long ioStart = DMLScript.OOC_STATISTICS ? System.nanoTime() : 0;
			ix.readFields(dis); // 1. Read Indexes
			mb.readFields(dis); // 2. Read Block
			if (DMLScript.OOC_STATISTICS)
				ioDuration = System.nanoTime() - ioStart;
		} catch (ClosedByInterruptException ignored) {
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		block.setDataUnsafe(new IndexedMatrixValue(ix, mb));

		if (DMLScript.OOC_STATISTICS) {
			Statistics.incrementOOCLoadFromDisk();
			Statistics.accumulateOOCLoadFromDiskTime(ioDuration);
		}
	}

	private void loadFromSource(BlockEntry block, SourceBlockDescriptor src) {
		if (src.format != Types.FileFormat.BINARY)
			throw new DMLRuntimeException("Unsupported format for source read: " + src.format);

		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(src.path);

		MatrixIndexes ix = new MatrixIndexes();
		MatrixBlock mb = new MatrixBlock();

		try(SequenceFile.Reader reader = new SequenceFile.Reader(job, SequenceFile.Reader.file(path))) {
			reader.seek(src.offset);
			if (!reader.next(ix, mb))
				throw new DMLRuntimeException("Failed to read source block at offset " + src.offset + " in " + src.path);
		}
		catch(IOException e) {
			throw new DMLRuntimeException(e);
		}

		block.setDataUnsafe(new IndexedMatrixValue(ix, mb));
	}

	private void evictTask(CloseableQueue<Tuple2<BlockEntry, CompletableFuture<Void>>> q) {
		long byteCtr = 0;

		while (!q.isFinished()) {
			// --- 1. WRITE PHASE ---
			int partitionId = _partitionCounter.getAndIncrement();

			LocalFileUtils.createLocalFileIfNotExist(_spillDir);

			String filename = _spillDir + "/stream_batch_part_" + partitionId;

			PartitionFile partFile = new PartitionFile(filename);
			_partitions.put(partitionId, partFile);
			partFile.incrementRefCount(); // Writer pin; released when partition closes

			FileOutputStream fos = null;
			CountableFastBufferedDataOutputStream dos = null;
			ConcurrentLinkedDeque<Tuple3<Long, Long, CompletableFuture<Void>>> waitingForFlush = null;

			try {
				fos = new FileOutputStream(filename);
				dos = new CountableFastBufferedDataOutputStream(fos);

				Tuple2<BlockEntry, CompletableFuture<Void>> tpl;
				waitingForFlush = new ConcurrentLinkedDeque<>();
				boolean closePartition = false;

				while((tpl = q.take()) != null) {
					long ioStart = DMLScript.OOC_STATISTICS || DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
					BlockEntry entry = tpl._1;
					CompletableFuture<Void> future = tpl._2;
					long wrote = writeOut(partitionId, entry, future, fos, dos, waitingForFlush);

					if(DMLScript.OOC_STATISTICS && wrote > 0) {
						Statistics.incrementOOCEvictionWrite();
						Statistics.accumulateOOCEvictionWriteTime(System.nanoTime() - ioStart);
					}

					byteCtr += wrote;
					if (byteCtr >= MAX_PARTITION_SIZE) {
						closePartition = true;
						byteCtr = 0;
						break;
					}

					if (DMLScript.OOC_LOG_EVENTS)
						OOCEventLog.onDiskWriteEvent(_evictCallerId, ioStart, System.nanoTime(), wrote);
				}

				if (!closePartition && q.close()) {
					while((tpl = q.take()) != null) {
						long ioStart = DMLScript.OOC_STATISTICS ? System.nanoTime() : 0;
						BlockEntry entry = tpl._1;
						CompletableFuture<Void> future = tpl._2;
						long wrote = writeOut(partitionId, entry, future, fos, dos, waitingForFlush);
						byteCtr += wrote;

						if(DMLScript.OOC_STATISTICS && wrote > 0) {
							Statistics.incrementOOCEvictionWrite();
							Statistics.accumulateOOCEvictionWriteTime(System.nanoTime() - ioStart);
						}

						if (DMLScript.OOC_LOG_EVENTS)
							OOCEventLog.onDiskWriteEvent(_evictCallerId, ioStart, System.nanoTime(), wrote);
					}
				}
			}
			catch(IOException | InterruptedException ex) {
				throw new DMLRuntimeException(ex);
			}
			catch(Exception ignored) {
			}
			finally {
				IOUtilFunctions.closeSilently(dos);
				IOUtilFunctions.closeSilently(fos);
				if(waitingForFlush != null)
					flushQueue(Long.MAX_VALUE, waitingForFlush);
				releasePartitionWriter(partitionId);
			}
		}
	}

	private long writeOut(int partitionId, BlockEntry entry, CompletableFuture<Void> future, FileOutputStream fos,
		CountableFastBufferedDataOutputStream dos, ConcurrentLinkedDeque<Tuple3<Long, Long, CompletableFuture<Void>>> flushQueue) throws IOException {
		String key = entry.getKey().toFileKey();
		boolean alreadySpilled = _spillLocations.containsKey(key);

		if (!alreadySpilled) {
			// 1. get the current file position. this is the offset.
			// flush any buffered data to the file
			//dos.flush();
			long offsetBefore = fos.getChannel().position() + dos.getCount();

			// 2. write indexes and block
			IndexedMatrixValue imv = (IndexedMatrixValue) entry.getDataUnsafe(); // Get data without requiring pin
			if(imv == null)
				return 0;
			imv.getIndexes().write(dos); // write Indexes
			imv.getValue().write(dos);

			long offsetAfter = fos.getChannel().position() + dos.getCount();
			flushQueue.offer(new Tuple3<>(offsetBefore, offsetAfter, future));

			// 3. create the spillLocation
			SpillLocation sloc = new SpillLocation(partitionId, offsetBefore);
			addSpillLocation(key, sloc);
			flushQueue(fos.getChannel().position(), flushQueue);

			return offsetAfter - offsetBefore;
		}
		return 0;
	}

	private void flushQueue(long offset, ConcurrentLinkedDeque<Tuple3<Long, Long, CompletableFuture<Void>>> flushQueue) {
		Tuple3<Long, Long, CompletableFuture<Void>> tmp;
		while ((tmp = flushQueue.peek()) != null && tmp._2() < offset) {
			flushQueue.poll();
			tmp._3().complete(null);
		}
	}

	private void addSpillLocation(String key, SpillLocation sloc) {
		synchronized(_spillLock) {
			SpillLocation existing = _spillLocations.putIfAbsent(key, sloc);
			if(existing == null) {
				PartitionFile partFile = _partitions.get(sloc.partitionId);
				if(partFile != null)
					partFile.incrementRefCount();
			}
		}
	}

	private void removeSpillLocation(String key) {
		synchronized(_spillLock) {
			SpillLocation sloc = _spillLocations.remove(key);
			if(sloc == null)
				return;

			PartitionFile partFile = _partitions.get(sloc.partitionId);
			if(partFile == null)
				return;

			int remaining = partFile.decrementRefCount();
			if(remaining == 0 && _partitions.remove(sloc.partitionId, partFile)) {
				try {
					_deleteExec.execute(() -> LocalFileUtils.deleteFileIfExists(partFile.filePath, true));
				}
				catch(RejectedExecutionException ignored) {
				}
			}
		}
	}

	private void releasePartitionWriter(int partitionId) {
		synchronized(_spillLock) {
			PartitionFile partFile = _partitions.get(partitionId);
			if(partFile == null)
				return;
			int remaining = partFile.decrementRefCount();
			if(remaining == 0 && _partitions.remove(partitionId, partFile)) {
				try {
					_deleteExec.execute(() -> LocalFileUtils.deleteFileIfExists(partFile.filePath, true));
				}
				catch(RejectedExecutionException ignored) {
				}
			}
		}
	}

	private int pinPartitionForRead(BlockKey key) {
		String fileKey = key.toFileKey();
		synchronized(_spillLock) {
			SpillLocation sloc = _spillLocations.get(fileKey);
			if(sloc == null)
				return -1;
			PartitionFile partFile = _partitions.get(sloc.partitionId);
			if(partFile == null)
				return -1;
			partFile.incrementRefCount();
			return sloc.partitionId;
		}
	}

	private void unpinPartitionForRead(int partitionId) {
		if(partitionId < 0)
			return;
		synchronized(_spillLock) {
			PartitionFile partFile = _partitions.get(partitionId);
			if(partFile == null)
				return;
			int remaining = partFile.decrementRefCount();
			if(remaining == 0 && _partitions.remove(partitionId, partFile)) {
				try {
					_deleteExec.execute(() -> LocalFileUtils.deleteFileIfExists(partFile.filePath, true));
				}
				catch(RejectedExecutionException ignored) {
				}
			}
		}
	}

	private class ReadTask implements Runnable, Comparable<ReadTask> {
		private final BlockEntry _block;
		private final CompletableFuture<BlockEntry> _future;
		private final long _sequence;
		private final int _pinnedPartitionId;
		private double _priority;

		private ReadTask(BlockEntry block, CompletableFuture<BlockEntry> future, long sequence, int pinnedPartitionId) {
			this._block = block;
			this._future = future;
			this._sequence = sequence;
			this._pinnedPartitionId = pinnedPartitionId;
			this._priority = 0;
		}

		private void addPriority(double delta) {
			_priority += delta;
		}

		@Override
		public void run() {
			_pendingReads.remove(_block.getKey(), this);
			try {
				long ioStart = DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
				loadFromDisk(_block);
				if (DMLScript.OOC_LOG_EVENTS)
					OOCEventLog.onDiskReadEvent(_readCallerId, ioStart, System.nanoTime(), _block.getSize());
				_future.complete(_block);
			} catch (Throwable e) {
				_future.completeExceptionally(e);
			} finally {
				unpinPartitionForRead(_pinnedPartitionId);
			}
		}

		@Override
		public int compareTo(ReadTask other) {
			int byPriority = Double.compare(other._priority, _priority);
			if (byPriority != 0)
				return byPriority;
			return Long.compare(_sequence, other._sequence);
		}
	}




	private static class SpillLocation {
		// structure of spillLocation: file, offset
		final int partitionId;
		final long offset;

		SpillLocation(int partitionId, long offset) {
			this.partitionId = partitionId;
			this.offset = offset;
		}
	}

	private static class PartitionFile {
		final String filePath;
		private final AtomicInteger refCount;

		PartitionFile(String filePath) {
			this.filePath = filePath;
			this.refCount = new AtomicInteger(0);
		}

		int incrementRefCount() {
			return refCount.incrementAndGet();
		}

		int decrementRefCount() {
			return refCount.decrementAndGet();
		}
	}

	private static class CountableFastBufferedDataOutputStream extends FastBufferedDataOutputStream {
		public CountableFastBufferedDataOutputStream(OutputStream out) {
			super(out);
		}

		public int getCount() {
			return _count;
		}
	}

	private static class SourceReadState implements SourceReadContinuation {
		final SourceReadRequest request;
		final Path[] paths;
		final AtomicLongArray filePositions;
		final AtomicIntegerArray completed;

		SourceReadState(SourceReadRequest request, Path[] paths, AtomicLongArray filePositions,
			AtomicIntegerArray completed) {
			this.request = request;
			this.paths = paths;
			this.filePositions = filePositions;
			this.completed = completed;
		}
	}
}
