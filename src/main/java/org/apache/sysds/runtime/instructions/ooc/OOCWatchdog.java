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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.runtime.ooc.cache.BlockEntry;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCCacheScheduler;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Watchdog to help debug OOC streams/tasks that never close.
 */
public final class OOCWatchdog {
	public static final boolean WATCH = false;
	private static final double PINNED_NEAR_LIMIT_RATIO = 0.9;
	private static final int TOP_PINNED_STREAMS = 5;
	private static final ConcurrentHashMap<String, Entry> OPEN = new ConcurrentHashMap<>();
	private static final ScheduledExecutorService EXEC = Executors.newSingleThreadScheduledExecutor(r -> {
		Thread t = new Thread(r, "TemporaryWatchdog");
		t.setDaemon(true);
		return t;
	});

	private static final long STALE_MS = TimeUnit.SECONDS.toMillis(10);
	private static final long SCAN_INTERVAL_MS = TimeUnit.SECONDS.toMillis(10);
	private static final long CACHE_SCAN_INTERVAL_MS = TimeUnit.SECONDS.toMillis(1);

	static {
		if(WATCH) {
			EXEC.scheduleAtFixedRate(OOCWatchdog::scan, SCAN_INTERVAL_MS, SCAN_INTERVAL_MS, TimeUnit.MILLISECONDS);
			EXEC.scheduleAtFixedRate(OOCWatchdog::scanCachePressure, CACHE_SCAN_INTERVAL_MS, CACHE_SCAN_INTERVAL_MS,
				TimeUnit.MILLISECONDS);
		}
	}

	private OOCWatchdog() {
		// no-op
	}

	public static void registerOpen(String id, String desc, String context, OOCStreamable<?> stream) {
		OPEN.put(id, new Entry(desc, context, System.currentTimeMillis(), stream));
	}

	public static void addEvent(String id, String eventMsg) {
		Entry e = OPEN.get(id);
		if(e != null)
			e.events.add(eventMsg);
	}

	public static void registerClose(String id) {
		OPEN.remove(id);
	}

	private static void scan() {
		long now = System.currentTimeMillis();
		for(Map.Entry<String, Entry> e : OPEN.entrySet()) {
			if(now - e.getValue().openedAt >= STALE_MS) {
				if(e.getValue().events.isEmpty() && !(e.getValue().stream instanceof CachingStream))
					continue; // Probably just a stream that has no consumer (remains to be checked why this can happen)
				System.err.println(
					"[TemporaryWatchdog] Still open after " + (now - e.getValue().openedAt) + "ms: " + e.getKey() +
						" (" + e.getValue().desc + ")" +
						(e.getValue().context != null ? " ctx=" + e.getValue().context : ""));
			}
		}
	}

	private static void scanCachePressure() {
		OOCCacheScheduler cache = OOCCacheManager.getCacheIfInitialized();
		if(cache == null)
			return;

		long hardLimit = cache.getHardLimit();
		if(hardLimit <= 0)
			return;
		long pinnedBytes = cache.getPinnedBytes();
		if(pinnedBytes < (long) (hardLimit * PINNED_NEAR_LIMIT_RATIO))
			return;

		Collection<BlockEntry> snapshot = cache.snapshot();
		if(snapshot.isEmpty())
			return;

		HashMap<Long, StreamPinStats> pinnedByStream = new HashMap<>();
		long pinnedBlocks = 0;
		for(BlockEntry entry : snapshot) {
			if(!entry.isPinned())
				continue;
			long streamId = entry.getKey().getStreamId();
			StreamPinStats stats = pinnedByStream.computeIfAbsent(streamId, sid -> new StreamPinStats());
			stats.bytes += entry.getSize();
			stats.blocks++;
			pinnedBlocks++;
		}

		if(pinnedByStream.isEmpty())
			return;

		ArrayList<Map.Entry<Long, StreamPinStats>> top = new ArrayList<>(pinnedByStream.entrySet());
		top.sort(Comparator.comparingLong((Map.Entry<Long, StreamPinStats> e) -> e.getValue().bytes).reversed());

		StringBuilder sb = new StringBuilder();
		sb.append("[WARN] OOCWatchdog: pinned memory near hard limit: ");
		sb.append(toMiB(pinnedBytes)).append("MiB / ").append(toMiB(hardLimit)).append("MiB (")
			.append(String.format("%.1f", 100.0 * pinnedBytes / hardLimit)).append("%)");
		sb.append(", pinned blocks=").append(pinnedBlocks).append(", top streams=[");

		int n = Math.min(TOP_PINNED_STREAMS, top.size());
		for(int i = 0; i < n; i++) {
			Map.Entry<Long, StreamPinStats> e = top.get(i);
			if(i > 0)
				sb.append("; ");
			sb.append(e.getKey()).append(": ").append(toMiB(e.getValue().bytes)).append("MiB (")
				.append(e.getValue().blocks).append(" blocks)");
		}
		sb.append("]");
		System.err.println(sb);
	}

	private static long toMiB(long bytes) {
		return bytes / (1024 * 1024);
	}

	private static class Entry {
		final String desc;
		final String context;
		final long openedAt;
		final OOCStreamable<?> stream;
		ConcurrentLinkedQueue<String> events;

		Entry(String desc, String context, long openedAt, OOCStreamable<?> stream) {
			this.desc = desc;
			this.context = context;
			this.openedAt = openedAt;
			this.stream = stream;
			this.events = new ConcurrentLinkedQueue<>();
		}
	}

	private static class StreamPinStats {
		long bytes;
		long blocks;
	}
}
