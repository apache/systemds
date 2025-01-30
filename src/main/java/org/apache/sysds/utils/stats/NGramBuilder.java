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

package org.apache.sysds.utils.stats;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class NGramBuilder<T, U> {

	public static <T, U> String toCSV(String[] columnNames, List<NGramEntry<T, U>> entries, Function<NGramEntry<T, U>, String> statsMapper) {
		StringBuilder builder = new StringBuilder(String.join(",", columnNames));
		builder.append("\n");

		for (NGramEntry<T, U> entry : entries) {
			builder.append(entry.getIdentifier().replace(",", ";"));
			builder.append(",");
			builder.append(entry.getCumStats());
			builder.append(",");

			if (statsMapper != null) {
				builder.append(statsMapper.apply(entry));
				builder.append(",");
			}

			builder.append(entry.getOccurrences());
			builder.append("\n");
		}

		return builder.toString();
	}

	public static <T, U> void toCSVStream(String[] columnNames, List<NGramEntry<T, U>> entries, Function<NGramEntry<T, U>, String> statsMapper, Consumer<String> lineConsumer) {
		StringBuilder builder = new StringBuilder(String.join(",", columnNames));
		builder.append("\n");
		lineConsumer.accept(builder.toString());
		builder.setLength(0);

		for (NGramEntry<T, U> entry : entries) {
			builder.append(entry.getIdentifier().replace(",", ";"));
			builder.append(",");
			builder.append(entry.getCumStats());
			builder.append(",");

			if (statsMapper != null) {
				builder.append(statsMapper.apply(entry));
				builder.append(",");
			}

			builder.append(entry.getOccurrences());
			builder.append("\n");
			lineConsumer.accept(builder.toString());
			builder.setLength(0);
		}
	}

	public static class NGramEntry<T, U> {
		private final String identifier;
		private final T[] entry;
		private U[] stats;
		private U cumStats;
		private long occurrences;
		private int offset;

		public NGramEntry(String identifier, T[] entry, U[] stats, U cumStats, int offset) {
			this.identifier = identifier;
			this.entry = entry;
			this.stats = stats;
			this.occurrences = 1;
			this.offset = offset;
			this.cumStats = cumStats;
		}

		public String getIdentifier() {
			return identifier;
		}

		public long getOccurrences() {
			return occurrences;
		}

		public U getStat(int index) {
			if (index < 0 || index >= entry.length)
				throw new ArrayIndexOutOfBoundsException("Index " + index + " is out of bounds");

			index = (index + offset) % entry.length;
			return stats[index];
		}

		public U getCumStats() {
			return cumStats;
		}

		public U[] getStats() {
			return stats;
		}

		public int getOffset() {
			return offset;
		}

		void setCumStats(U cumStats) {
			this.cumStats = cumStats;
		}

		public T get(int index) {
			if (index < 0 || index >= entry.length)
				throw new ArrayIndexOutOfBoundsException("Index " + index + " is out of bounds");

			index = (index + offset) % entry.length;
			return entry[index];
		}

		private NGramEntry<T, U> increment() {
			occurrences++;
			return this;
		}

		@SuppressWarnings("unused")
		private NGramEntry<T, U> add(NGramEntry<T, U> entry) {
			return add(entry.occurrences);
		}

		private NGramEntry<T, U> add(long n) {
			occurrences += n;
			return this;
		}
	}

	private final T[] currentNGram;
	private final U[] currentStats;
	private int currentIndex = 0;
	private int currentSize = 0;
	private final Function<T, String> idGenerator;
	private final BiFunction<U, U, U> statsMerger;
	private final ConcurrentHashMap<String, NGramEntry<T, U>> nGrams;

	@SuppressWarnings("unchecked")
	public NGramBuilder(Class<T> clazz, Class<U> clazz2, int size, Function<T, String> idGenerator, BiFunction<U, U, U> statsMerger) {
		currentNGram = (T[]) Array.newInstance(clazz, size);
		currentStats = (U[]) Array.newInstance(clazz2, size);
		this.idGenerator = idGenerator;
		this.nGrams = new ConcurrentHashMap<>();
		this.statsMerger = statsMerger;
	}

	public int getSize() {
		return currentNGram.length;
	}

	public synchronized void merge(NGramBuilder<T, U> builder) {
		builder.nGrams.forEach((k, v) -> nGrams.merge(k, v, (v1, v2) ->
		{
			v1.add(v2.occurrences);
			v1.setCumStats(statsMerger.apply(v1.getCumStats(), v2.getCumStats()));
			int index1 = v1.offset;
			int index2 = v2.offset;
			U[] stats1 = v1.getStats();
			U[] stats2 = v2.getStats();

			for (int i = 0; i < stats1.length; i++) {
				stats1[index1] = statsMerger.apply(stats1[index1], stats2[index2]);
				index1 = (index1 + 1) % stats1.length;
				index2 = (index2 + 1) % stats2.length;
			}

			return v1;
		}));
	}

	public synchronized void append(T element, U stat) {
		currentNGram[currentIndex] = element;
		currentStats[currentIndex] = stat;
		currentIndex = (currentIndex + 1) % currentNGram.length;

		if (currentSize < currentNGram.length)
			currentSize++;

		if (currentSize == currentNGram.length) {
			StringBuilder builder = new StringBuilder(currentNGram.length);
			builder.append("(");

			for (int i = 0; i < currentNGram.length; i++) {
				int actualIndex = (i + currentIndex) % currentSize;
				builder.append(idGenerator.apply(currentNGram[actualIndex]));

				if (i != currentNGram.length - 1)
					builder.append(", ");
			}

			builder.append(")");

			registerElement(builder.toString(), stat);
		}
	}

	public synchronized List<NGramEntry<T, U>> getTopK(int k) {
		return nGrams.entrySet().stream()
				.sorted(Comparator.comparingLong((Map.Entry<String, NGramEntry<T, U>> v) -> v.getValue().occurrences).reversed())
				.map(Map.Entry::getValue)
				.limit(k)
				.collect(Collectors.toList());
	}

	public synchronized List<NGramEntry<T, U>> getTopK(int k, Comparator<NGramEntry<T, U>> comparator, boolean reversed) {
		return nGrams.entrySet().stream()
				.sorted((e1, e2) -> reversed ? comparator.compare(e2.getValue(), e1.getValue()) : comparator.compare(e1.getValue(), e2.getValue()))
				.map(Map.Entry::getValue)
				.limit(k)
				.collect(Collectors.toList());
	}

	public synchronized void clearCurrentRecording() {
		currentIndex = 0;
		currentSize = 0;
	}

	private synchronized void registerElement(String id, U stat) {
		nGrams.compute(id, (key, entry) ->  {
			if (entry == null) {
				U cumStat = currentStats[0];

				for (int i = 1; i < currentStats.length; i++) {
					cumStat = statsMerger.apply(currentStats[i], cumStat);
				}

				entry = new NGramEntry<T, U>(id, Arrays.copyOf(currentNGram, currentNGram.length), Arrays.copyOf(currentStats, currentStats.length), cumStat, currentIndex);
			} else {
				entry.increment();
				U[] stats = entry.getStats();
				U cumStat = null;

				int mCurrentIndex = currentIndex;
				int mIndexEntry = entry.offset;

				for (int i = 0; i < stats.length; i++) {
					stats[mIndexEntry] = statsMerger.apply(stats[mIndexEntry], currentStats[mCurrentIndex]);
					if (i == 0) {
						cumStat = stats[mIndexEntry];
					} else {
						cumStat = statsMerger.apply(stats[mIndexEntry], cumStat);
					}

					mCurrentIndex = (mCurrentIndex + 1) % stats.length;
					mIndexEntry = (mIndexEntry + 1) % stats.length;
				}

				entry.setCumStats(cumStat);
			}

			return entry;
		});
	}

}
