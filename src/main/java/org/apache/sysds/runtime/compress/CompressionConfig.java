package org.apache.sysds.runtime.compress;

import java.util.HashMap;
import java.util.Map;

/**
 * Immutable configuration for compression in federated operations.
 * Uses the Builder pattern for flexible, readable configuration.
 *
 * Usage example:
 *   CompressionConfig config = CompressionConfig.builder()
 *       .enable(true)
 *       .withType(CompressionType.TOPK)
 *       .withSparsity(0.01)
 *       .build();
 *
 *
 */
public class CompressionConfig {

    private final boolean enabled;
    private final CompressionType type;
    private final Map<String, Object> parameters;

    private CompressionConfig(Builder builder) {
        this.enabled = builder.enabled;
        this.type = builder.enabled ? builder.type : CompressionType.NONE;
        this.parameters = new HashMap<>(builder.parameters);
    }

    public boolean isEnabled() { return enabled; }
    public CompressionType getType() { return type; }
    public Map<String, Object> getParameters() { return new HashMap<>(parameters); }

    /** Convenience getter for sparsity parameter (TopK) */
    public double getSparsity() {
        return (double) parameters.getOrDefault("sparsity", 0.01);
    }

    /** Convenience getter for bits parameter (Quantization) */
    public int getBits() {
        return (int) parameters.getOrDefault("bits", 4);
    }

    @Override
    public String toString() {
        return String.format("CompressionConfig[enabled=%s, type=%s, params=%s]",
            enabled, type.getId(), parameters);
    }

    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private boolean enabled = false;
        private CompressionType type = CompressionType.NONE;
        private final Map<String, Object> parameters = new HashMap<>();

        public Builder enable(boolean enabled) {
            this.enabled = enabled;
            return this;
        }

        public Builder withType(CompressionType type) {
            this.type = type;
            return this;
        }

        public Builder withParameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        /** Shorthand for TopK sparsity ratio */
        public Builder withSparsity(double sparsity) {
            return withParameter("sparsity", sparsity);
        }

        /** Shorthand for quantization bit width */
        public Builder withBits(int bits) {
            return withParameter("bits", bits);
        }

        public CompressionConfig build() {
            return new CompressionConfig(this);
        }
    }
}