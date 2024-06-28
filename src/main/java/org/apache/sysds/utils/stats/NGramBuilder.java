package org.apache.sysds.utils.stats;


import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class NGramBuilder<T, U> {
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

		U[] getStats() {
			return stats;
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
	//private final Function<NGramEntry<T, U>, U> statsMaintainer;
	private final ConcurrentHashMap<String, NGramEntry<T, U>> nGrams;
	private final NGramBuilder<T, U> smallerNGramBuilder;

	@SuppressWarnings("unchecked")
	public NGramBuilder(Class<T> clazz, Class<U> clazz2, int maxSize, int minSize, Function<T, String> idGenerator, BiFunction<U, U, U> statsMerger) {
		currentNGram = (T[]) Array.newInstance(clazz, maxSize);
		currentStats = (U[]) Array.newInstance(clazz2, maxSize);
		this.idGenerator = idGenerator;
		this.nGrams = new ConcurrentHashMap<>();
		this.statsMerger = statsMerger;

		if (maxSize > minSize) {
			smallerNGramBuilder = new NGramBuilder<>(clazz, clazz2, maxSize-1, minSize, idGenerator, statsMerger);
		} else {
			smallerNGramBuilder = null;
		}
	}

	public synchronized NGramBuilder<T, U> getChild() {
		return smallerNGramBuilder;
	}

	public synchronized void merge(NGramBuilder<T, U> builder) {
		builder.nGrams.forEach((k, v) -> nGrams.merge(k, v, (v1, v2) ->
		{
			v1.add(v2.occurrences);
			U[] stats1 = v1.getStats();
			U[] stats2 = v2.getStats();

			for (int i = 0; i < stats1.length; i++) {
				stats1[i] = statsMerger.apply(stats1[i], stats2[i]);
			}

			return v1;
		}));
	}

	public synchronized void append(T element, U stat) {
		if (smallerNGramBuilder != null)
			smallerNGramBuilder.append(element, stat);

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

				for (int i = 0; i < stats.length; i++) {
					stats[i] = statsMerger.apply(stats[i], currentStats[i]);
					if (i == 0) {
						cumStat = stats[0];
					} else {
						cumStat = statsMerger.apply(stats[i], cumStat);
					}
				}

				entry.setCumStats(cumStat);
			}

			return entry;
		});
	}

}
