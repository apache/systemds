package org.apache.sysds.utils.stats;


import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;

public class NGramBuilder<T> {
    public static class NGramEntry<T> {
        private final String identifier;
        private final T[] entry;
        private long occurrences;
        private int offset;

        public NGramEntry(String identifier, T[] entry, int offset) {
            this.identifier = identifier;
            this.entry = entry;
            this.occurrences = 1;
            this.offset = offset;
        }

        public String getIdentifier() {
            return identifier;
        }

        public long getOccurrences() {
            return occurrences;
        }

        public T get(int index) {
            if (index < 0 || index >= entry.length)
                throw new ArrayIndexOutOfBoundsException("Index " + index + " is out of bounds");

            index = (index + offset) % entry.length;
            return entry[index];
        }

        private NGramEntry<T> increment() {
            occurrences++;
            return this;
        }

        private NGramEntry<T> add(NGramEntry<T> entry) {
            return add(entry.occurrences);
        }

        private NGramEntry<T> add(long n) {
            occurrences += n;
            return this;
        }
    }

    private final T[] currentNGram;
    private int currentIndex = 0;
    private int currentSize = 0;
    private final Function<T, String> idGenerator;
    private final ConcurrentHashMap<String, NGramEntry<T>> nGrams;

    @SuppressWarnings("unchecked")
    public NGramBuilder(Class<T> clazz, int maxSize, Function<T, String> idGenerator) {
        currentNGram = (T[]) Array.newInstance(clazz, maxSize);
        this.idGenerator = idGenerator;
        this.nGrams = new ConcurrentHashMap<>();
    }

    public synchronized void merge(NGramBuilder<T> builder) {
        builder.nGrams.forEach((k, v) -> nGrams.merge(k, v, (v1, v2) -> v1.add(v2.occurrences)));
    }

    public synchronized void append(T element) {
        currentNGram[currentIndex] = element;
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

            registerElement(builder.toString());
        }
    }

    public List<NGramEntry<T>> getTopK(int k) {
        return nGrams.entrySet().stream()
                .sorted(Comparator.comparingLong((Map.Entry<String, NGramEntry<T>> v) -> v.getValue().occurrences).reversed())
                .map(Map.Entry::getValue)
                .limit(k)
                .collect(Collectors.toList());
    }

    private void registerElement(String id) {
        nGrams.compute(id, (key, entry) ->  entry == null ? new NGramEntry<T>(id, Arrays.copyOf(currentNGram, currentNGram.length), currentIndex) : entry.increment());
    }

}
