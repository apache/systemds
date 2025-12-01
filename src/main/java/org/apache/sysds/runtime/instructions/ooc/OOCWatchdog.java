package org.apache.sysds.runtime.instructions.ooc;

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
	private static final ConcurrentHashMap<String, Entry> OPEN = new ConcurrentHashMap<>();
	private static final ScheduledExecutorService EXEC =
		Executors.newSingleThreadScheduledExecutor(r -> {
			Thread t = new Thread(r, "TemporaryWatchdog");
			t.setDaemon(true);
			return t;
		});

	private static final long STALE_MS = TimeUnit.SECONDS.toMillis(10);
	private static final long SCAN_INTERVAL_MS = TimeUnit.SECONDS.toMillis(10);

	static {
		EXEC.scheduleAtFixedRate(OOCWatchdog::scan, SCAN_INTERVAL_MS, SCAN_INTERVAL_MS, TimeUnit.MILLISECONDS);
	}

	private OOCWatchdog() {
		// no-op
	}

	public static void registerOpen(String id, String desc, String context, OOCStream<?> stream) {
		OPEN.put(id, new Entry(desc, context, System.currentTimeMillis(), stream));
	}

	public static void addEvent(String id, String eventMsg) {
		Entry e = OPEN.get(id);
		if (e != null)
			e.events.add(eventMsg);
	}

	public static void registerClose(String id) {
		OPEN.remove(id);
	}

	private static void scan() {
		long now = System.currentTimeMillis();
		for (Map.Entry<String, Entry> e : OPEN.entrySet()) {
			if (now - e.getValue().openedAt >= STALE_MS) {
				if (e.getValue().events.isEmpty())
					continue; // Probably just a stream that has no consumer (remains to be checked why this can happen)
				System.err.println("[TemporaryWatchdog] Still open after " + (now - e.getValue().openedAt) + "ms: "
					+ e.getKey() + " (" + e.getValue().desc + ")"
					+ (e.getValue().context != null ? " ctx=" + e.getValue().context : ""));
			}
		}
	}

	private static class Entry {
		final String desc;
		final String context;
		final long openedAt;
		final OOCStream<?> stream;
		ConcurrentLinkedQueue<String> events;

		Entry(String desc, String context, long openedAt, OOCStream<?> stream) {
			this.desc = desc;
			this.context = context;
			this.openedAt = openedAt;
			this.stream = stream;
			this.events = new ConcurrentLinkedQueue<>();
		}
	}
}
