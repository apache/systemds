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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.ooc.stats.OOCEventLog;
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
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class OOCMatrixIOHandler implements OOCIOHandler {
	private static final int WRITER_SIZE = 2;
	private static final long OVERFLOW = 8192 * 1024;
	private static final long MAX_PARTITION_SIZE = 8192 * 8192;

	private final String _spillDir;
	private final ThreadPoolExecutor _writeExec;
	private final ThreadPoolExecutor _readExec;

	// Spill related structures
	private final ConcurrentHashMap<String, SpillLocation> _spillLocations =  new ConcurrentHashMap<>();
	private final ConcurrentHashMap<Integer, PartitionFile> _partitions = new ConcurrentHashMap<>();
	private final AtomicInteger _partitionCounter = new AtomicInteger(0);
	private final CloseableQueue<Tuple2<BlockEntry, CompletableFuture<Void>>>[] _q;
	private final AtomicLong _wCtr;
	private final AtomicBoolean _started;

	private final int _evictCallerId = OOCEventLog.registerCaller("write");
	private final int _readCallerId = OOCEventLog.registerCaller("read");

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
			5,
			5,
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
		try {
			_readExec.submit(() -> {
				try {
					long ioStart = DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
					loadFromDisk(block);
					if (DMLScript.OOC_LOG_EVENTS)
						OOCEventLog.onDiskReadEvent(_readCallerId, ioStart, System.nanoTime(), block.getSize());
					future.complete(block);
				} catch (Throwable e) {
					future.completeExceptionally(e);
				}
			});
		} catch (RejectedExecutionException e) {
			future.completeExceptionally(e);
		}
		return future;
	}

	@Override
	public CompletableFuture<Boolean> scheduleDeletion(BlockEntry block) {
		// TODO
		return CompletableFuture.completedFuture(true);
	}


	private void loadFromDisk(BlockEntry block) {
		String key = block.getKey().toFileKey();

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
			long ioStart = DMLScript.STATISTICS ? System.nanoTime() : 0;
			ix.readFields(dis); // 1. Read Indexes
			mb.readFields(dis); // 2. Read Block
			if (DMLScript.STATISTICS)
				ioDuration = System.nanoTime() - ioStart;
		} catch (ClosedByInterruptException ignored) {
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		block.setDataUnsafe(new IndexedMatrixValue(ix, mb));

		if (DMLScript.STATISTICS) {
			Statistics.incrementOOCLoadFromDisk();
			Statistics.accumulateOOCLoadFromDiskTime(ioDuration);
		}
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
					long ioStart = DMLScript.STATISTICS || DMLScript.OOC_LOG_EVENTS ? System.nanoTime() : 0;
					BlockEntry entry = tpl._1;
					CompletableFuture<Void> future = tpl._2;
					long wrote = writeOut(partitionId, entry, future, fos, dos, waitingForFlush);

					if(DMLScript.STATISTICS && wrote > 0) {
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
						long ioStart = DMLScript.STATISTICS ? System.nanoTime() : 0;
						BlockEntry entry = tpl._1;
						CompletableFuture<Void> future = tpl._2;
						long wrote = writeOut(partitionId, entry, future, fos, dos, waitingForFlush);
						byteCtr += wrote;

						if(DMLScript.STATISTICS && wrote > 0) {
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
			catch(Exception e) {
				// TODO
			}
			finally {
				IOUtilFunctions.closeSilently(dos);
				IOUtilFunctions.closeSilently(fos);
				if(waitingForFlush != null)
					flushQueue(Long.MAX_VALUE, waitingForFlush);
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
			imv.getIndexes().write(dos); // write Indexes
			imv.getValue().write(dos);

			long offsetAfter = fos.getChannel().position() + dos.getCount();
			flushQueue.offer(new Tuple3<>(offsetBefore, offsetAfter, future));

			// 3. create the spillLocation
			SpillLocation sloc = new SpillLocation(partitionId, offsetBefore);
			_spillLocations.put(key, sloc);
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

		PartitionFile(String filePath) {
			this.filePath = filePath;
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
}
