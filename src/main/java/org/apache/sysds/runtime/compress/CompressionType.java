package org.apache.sysds.runtime.compress;

/**
 * Enumeration of supported compression techniques for federated learning.
 * Used for configuration, serialization, and technique selection.
 *
 * @author Nirvan C. Udaysingh Jhurree
 */
public enum CompressionType {

    /** TopK sparsification: keep largest-magnitude elements only */
    TOPK("topk", "Top-K Sparsification"),

    /** Probabilistic quantization: reduce precision with stochastic rounding */
    PROBABILISTIC_QUANTIZATION("prob_quant", "Probabilistic Quantization"),

    /** 1-bit compressed sensing: sign-only transmission + iterative reconstruction */
    ONE_BIT_CS("1bit_cs", "1-Bit Compressed Sensing"),

    /** No compression (passthrough) */
    NONE("none", "No Compression");

    private final String id;
    private final String description;

    CompressionType(String id, String description) {
        this.id = id;
        this.description = description;
    }

    public String getId() { return id; }
    public String getDescription() { return description; }

    /** Parse from string identifier (case-insensitive) */
    public static CompressionType fromString(String text) {
        for (CompressionType type : CompressionType.values()) {
            if (type.id.equalsIgnoreCase(text)) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown compression type: " + text);
    }
}