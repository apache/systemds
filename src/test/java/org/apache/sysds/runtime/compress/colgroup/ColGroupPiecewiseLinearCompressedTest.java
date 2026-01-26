package org.apache.sysds.runtime.compress.colgroup;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.indexes.ArrayIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ColGroupPiecewiseLinearCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;

import org.junit.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Arrays;
import java.util.List;

import static org.apache.sysds.runtime.compress.colgroup.ColGroupFactory.computeBreakpoints;
import static org.junit.Assert.*;

/**
 * Tests für PiecewiseLinearColGroupCompressed, fokussiert auf:
 * - Konstruktor / create(...)
 * - decompressToDenseBlock(...)
 */
//TODO Fix
public class ColGroupPiecewiseLinearCompressedTest {
    @Test
    public void testComputeBreakpoints_uniformColumn() {
		CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(1e-3);
        double[] column = {1.0, 1.0, 1.0, 1.0, 1.0}; // ← Test-spezifisch
        List<Integer> breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0), breaks); // Erwartet: keine Breaks
    }

    @Test
    public void testComputeBreakpoints_linearIncreasing() {
		CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(1e-3);
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0}; // ← andere column
        List<Integer> breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0, 2), breaks); // Erwartet

    }

    @Test
    public void testComputeBreakpoints_highLoss_uniform() {
		CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(1.0); // ← andere Loss
        double[] column = {1.0, 1.0, 1.0, 1.0, 1.0};
        List<Integer> breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0), breaks);
    }

    @Test
    public void testComputeBreakpoints_noLoss_linear() {
		CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(0.0);
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0};
        List<Integer> breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0, 1, 2, 3), breaks); // bei 0 Loss alle Breaks
    }


}
