package org.apache.sysds.runtime.compress.colgroup;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ColGroupPiecewiseLinearCompressed;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;
import java.util.Arrays;
import java.util.List;
import static org.apache.sysds.runtime.compress.colgroup.ColGroupFactory.*;
import static org.junit.Assert.*;


public class ColGroupPiecewiseLinearCompressedTest {


    @Test
    public void testComputeBreakpoints_uniformColumn() {
        CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(1e-3);
        double[] column = {1.0, 1.0, 1.0, 1.0, 1.0}; // ← Test-spezifisch
        List<Integer> breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0, 5), breaks); // Erwartet: keine Breaks
    }

    @Test
    public void testComputeBreakpoints_linearIncreasing() {
        CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(1e-3);
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0}; // ← andere column
        List<Integer> breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0, 5), breaks); // Erwartet

    }

    @Test
    public void testComputeBreakpoints_highLoss_uniform() {
        CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(10000.0);
        double[] column = {1.0, 1.0, 1.0, 1.0, 1.0};
        List<Integer> breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0, 5), breaks);
    }

    @Test
    public void testComputeBreakpoints_twoSegments() {
        CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(1e-3);
        // {1,1,1, 2,2,2} → 2 Segmente → [0,3,6]
        double[] column = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
        var breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0, 3, 6), breaks);
    }

    @Test
    public void testComputeBreakpoints_noLoss_linear() {
        CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(0.0);
        //cs.setPiecewiseTargetLoss(0.0);
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0};
        List<Integer> breaks = computeBreakpoints(cs, column);
        assertEquals(Arrays.asList(0, 5), breaks); // bei 0 Loss alle Breaks
    }

    @Test
    public void testComputeBreakpointsLambda_const() {
        double[] column = {1.0, 1.0, 1.0, 1.0, 1.0};
        List<Integer> breaks = computeBreakpointsLambda(column, 5.0);
        assertEquals(Arrays.asList(0, 5), breaks);

        breaks = computeBreakpointsLambda(column, 0.01);
        assertEquals(Arrays.asList(0, 5), breaks);
    }

    @Test
    public void testComputeBreakpointsLambda_twoSegments() {
        double[] column = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};  // 6 Werte

        // mit kleinem lambda -> viele Segmente (kostenlos fast)
        List<Integer> breaks = computeBreakpointsLambda(column, 0.01);
        assertTrue(breaks.contains(3));
        assertEquals(3, breaks.size());
        assertEquals(Arrays.asList(0, 3, 6), breaks);

        // mit großem lambda entspricht nur ein Segment
        breaks = computeBreakpointsLambda(column, 1000.0);
        assertEquals(Arrays.asList(0, 6), breaks);
    }

    @Test
    public void testComputeBreakpointsLambda_jumpWithTrend() {
        double[] column = {0.0, 1.0, 2.0, 10.0, 11.0, 12.0};

        // grobe Segmentanpassung: ein Segment pro „Abschnitt“
        List<Integer> breaks = computeBreakpointsLambda(column, 0.5);
        assertEquals(Arrays.asList(0, 3, 6), breaks);

        // nur ein Segment, wenn lambda sehr groß
        breaks = computeBreakpointsLambda(column, 100.0);
        assertEquals(Arrays.asList(0, 6), breaks);
    }

    @Test
    public void testComputeBreakpointsLambda_linear() {
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

        List<Integer> breaks = computeBreakpointsLambda(column, 1.0);
        assertEquals(Arrays.asList(0, 6), breaks);

        // mit sehr kleinem lambda: wir prüfen nur, dass die Grenzen vernünftig sind
        breaks = computeBreakpointsLambda(column, 0.001);
        assertTrue(breaks.size() >= 2);
        assertTrue(breaks.get(0) == 0);
        assertTrue(breaks.get(breaks.size() - 1) == column.length);
    }

    @Test
    public void testComputeBreakpointsLambda_edge_lambdaVerySmall() {
        double[] column = {1.0, 1.1, 1.0, 1.1, 1.0};

        List<Integer> breaks = computeBreakpointsLambda(column, 0.001);
        assertNotNull(breaks);
        assertFalse(breaks.isEmpty());
        assertEquals(0, (int) breaks.get(0));
        assertEquals(column.length, (int) breaks.get(breaks.size() - 1));

        // Prüfe, dass die Liste sortiert ist
        for (int i = 1; i < breaks.size(); i++) {
            assertTrue(breaks.get(i) >= breaks.get(i - 1));
        }
    }

    @Test
    public void testComputeBreakpointsLambda_edge_lambdaVeryLarge() {
        double[] column = {1.0, 2.0, 1.5, 2.5, 1.8};

        List<Integer> breaks = computeBreakpointsLambda(column, 1000.0);
        assertEquals(Arrays.asList(0, 5), breaks);
    }

    @Test
    public void testComputeSegmentCost_emptyOrSingle() {
        double[] column = {10.0, 20.0, 30.0};

        // 0 Elemente (leer)
        assertEquals(0.0, computeSegmentCost(column, 0, 0), 1e-10);
        assertEquals(0.0, computeSegmentCost(column, 1, 1), 1e-10);

        // 1 Element → Regressionsgerade ist nicht eindeutig definiert, aber SSE=0
        assertEquals(0.0, computeSegmentCost(column, 0, 1), 1e-10);
        assertEquals(0.0, computeSegmentCost(column, 1, 2), 1e-10);
        assertEquals(0.0, computeSegmentCost(column, 2, 3), 1e-10);
    }

    @Test
    public void testComputeSegmentCost_twoConstantPoints() {
        double[] column = {5.0, 5.0, 1.0, 1.0};

        // Zwei identische Punkte (konstant) → SSE = 0
        double sse = computeSegmentCost(column, 0, 2);
        assertEquals(0.0, sse, 1e-10);
    }

    @Test
    public void testComputeSegmentCost_twoDifferentPoints() {
        double[] column = {0.0, 2.0, 1.0, 3.0};

        // Zwei Punkte: (0,0) und (1,2) → Gerade y = 2*x, Fehler = 0
        double sse = computeSegmentCost(column, 0, 2);
        assertEquals(0.0, sse, 1e-10);

        // Zwei Punkte: (2,1) und (3,3) → Gerade y = 2*x - 3, Fehler = 0
        sse = computeSegmentCost(column, 2, 4);
        assertEquals(0.0, sse, 1e-10);
    }

    @Test
    public void testComputeSegmentCost_constantThree() {
        double[] column = {0.0, 0.0, 0.0};
        double sse = computeSegmentCost(column, 0, 3);
        assertEquals(0.0, sse, 1e-10);
    }

    @Test
    public void testComputeSegmentCost_consistent_with_regression() {
        double[] column = {0.0, 2.0, 0.0, 4.0, 0.0, 6.0};

        int start = 0, end = 3;
        double[] ab = regressSegment(column, start, end);
        double slope = ab[0], intercept = ab[1];
        double sse_hand = 0.0;
        for (int i = start; i < end; i++) {
            double yhat = slope * i + intercept;
            double diff = column[i] - yhat;
            sse_hand += diff * diff;
        }

        double sse = computeSegmentCost(column, start, end);
        assertEquals(sse_hand, sse, 1e-10);
    }

    @Test
    public void testComputeTotalSSE_emptyBreaks() {
        double[] column = {1.0, 2.0, 3.0};
        List<Integer> breaks = Arrays.asList(); // leer → keine Segmente
        double total = computeTotalSSE(column, breaks);

        // 0 Segmente → Summe über 0 Segmente = 0
        assertEquals(0.0, total, 1e-10);
    }

    @Test
    public void testComputeTotalSSE_singleSegment_all() {
        double[] column = {1.0, 2.0, 3.0};
        List<Integer> breaks = Arrays.asList(0, 3); // ein Segment [0,3)

        double total = computeTotalSSE(column, breaks);
        double expected = computeSegmentCost(column, 0, 3);

        // Ergebnis muss exakt das gleiche wie der SSE des gesamten Segments sein
        assertEquals(expected, total, 1e-10);
    }

    @Test
    public void testComputeTotalSSE_twoSegments() {
        // Beispiel: [0,0,0] und [1,1,1] (jeweils konstant)
        double[] column = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
        List<Integer> breaks = Arrays.asList(0, 3, 6); // zwei Segmente

        double total = computeTotalSSE(column, breaks);
        double sse1 = computeSegmentCost(column, 0, 3); // [0,0,0] → SSE = 0
        double sse2 = computeSegmentCost(column, 3, 6); // [1,1,1] → SSE = 0

        // da beide Segmente konstant sind, muss totalSSE = 0 sein
        assertEquals(0.0, total, 1e-10);
        assertEquals(sse1 + sse2, total, 1e-10);
    }

    @Test
    public void testComputeTotalSSE_threeSegments() {
        // Ein Segment mit drei identischen Werten, zwei Segmente mit jeweils zwei Werten
        double[] column = {1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0};
        List<Integer> breaks = Arrays.asList(0, 3, 5, 7);

        // Segment [0,3): konstant 1.0 → SSE = 0
        double sse1 = computeSegmentCost(column, 0, 3); // 0

        // Segment [3,5): [2,2] → SSE = 0
        double sse2 = computeSegmentCost(column, 3, 5); // 0

        // Segment [5,7): [3,3] → SSE = 0
        double sse3 = computeSegmentCost(column, 5, 7); // 0

        double total = computeTotalSSE(column, breaks);
        assertEquals(0.0, total, 1e-10);
        assertEquals(sse1 + sse2 + sse3, total, 1e-10);
    }

    @Test
    public void testComputeTotalSSE_gapStartEnd() {
        double[] column = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        List<Integer> breaks = Arrays.asList(2, 5, 8);

        double total = computeTotalSSE(column, breaks);
        double sse1 = computeSegmentCost(column, 2, 5);
        double sse2 = computeSegmentCost(column, 5, 8);

        assertEquals(sse1 + sse2, total, 1e-10);

    }

    @Test
    public void testComputeTotalSSE_oneSegment_identical() {
        double[] column = {1.0, 2.0, 3.0, 4.0, 5.0};
        double sseTotal = computeSegmentCost(column, 0, 5);

        List<Integer> breaks = Arrays.asList(0, 5);
        double total = computeTotalSSE(column, breaks);

        assertEquals(sseTotal, total, 1e-10);
    }

    @Test
    public void testComputeTotalSSE_nonConstant() {
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0};
        List<Integer> breaks = Arrays.asList(0, 2, 5);

        double total = computeTotalSSE(column, breaks);
        double sse1 = computeSegmentCost(column, 0, 2);
        double sse2 = computeSegmentCost(column, 2, 5);

        assertTrue(total >= 0.0);
        assertEquals(sse1 + sse2, total, 1e-10);
    }

    @Test
    public void testComputeTotalSSE_edgeCases() {
        double[] columnEmpty = {};
        List<Integer> breaksEmpty = Arrays.asList(0, 0);
        assertEquals(0.0, computeTotalSSE(columnEmpty, breaksEmpty), 1e-10);

        double[] columnOne = {42.0};
        List<Integer> breaksOne = Arrays.asList(0, 1);
        double total = computeTotalSSE(columnOne, breaksOne);
        assertEquals(0.0, total, 1e-10);
    }

    @Test
    public void testRegressSegment_empty() {
        double[] column = {1.0, 2.0, 3.0};
        double[] result = regressSegment(column, 0, 0);
        assertEquals(0.0, result[0], 1e-10);
        assertEquals(0.0, result[1], 1e-10);
    }

    @Test
    public void testRegressSegment_singlePoint() {
        double[] column = {1.0, 2.0, 3.0};
        double[] result = regressSegment(column, 1, 2);

        assertEquals(0.0, result[0], 1e-10);
        assertEquals(2.0, result[1], 1e-10);
    }

    @Test
    public void testRegressSegment_twoIdentical() {
        double[] column = {5.0, 5.0, 1.0, 1.0};
        double[] result = regressSegment(column, 0, 2);

        assertEquals(0.0, result[0], 1e-10);
        assertEquals(5.0, result[1], 1e-10);
    }

    @Test
    public void testRegressSegment_twoPoints() {
        double[] column = {0.0, 2.0};
        double[] result = regressSegment(column, 0, 2);

        assertEquals(2.0, result[0], 1e-10);
        assertEquals(0.0, result[1], 1e-10);
    }

    @Test
    public void testRegressSegment_twoPoints_offset() {

        double[] column = {1.0, 3.0, 5.0, 7.0};
        double[] result = regressSegment(column, 2, 4);

        assertEquals(2.0, result[0], 1e-10);
        assertEquals(1.0, result[1], 1e-10);
    }

    @Test
    public void testRegressSegment_constant() {
        double[] column = {3.0, 3.0, 3.0, 3.0};
        double[] result = regressSegment(column, 0, 4);

        assertEquals(0.0, result[0], 1e-10);
        assertEquals(3.0, result[1], 1e-10);
    }

    @Test
    public void testRegressSegment_linear() {
        double[] column = new double[4];
        double a = 1.5, b = 2.0;
        for (int i = 0; i < 4; i++) {
            column[i] = a * i + b;
        }

        double[] result = regressSegment(column, 0, 4);

        assertEquals(a, result[0], 1e-10);
        assertEquals(b, result[1], 1e-10);
    }

    @Test
    public void testRegressSegment_denomZero() {
        double[] column = {10.0};
        double[] result = regressSegment(column, 0, 1);

        assertEquals(0.0, result[0], 1e-10);
        assertEquals(10.0, result[1], 1e-10);
    }

    @Test
    public void testCompressPiecewiseLinearFunctional_const() {
        // 1. MatrixBlock mit einer konstanten Spalte erzeugen
        int nrows = 20, ncols = 1;
        MatrixBlock in = new MatrixBlock(nrows, ncols, false);
        for (int r = 0; r < nrows; r++)
            in.set(r, 0, 1.0);
        // 2. colIndexes für Spalte 0
        IColIndex colIndexes = ColIndexFactory.create(new int[]{0});
        // 3. CompressionSettings mit TargetLoss
        CompressionSettings cs = new CompressionSettingsBuilder().create();
        cs.setPiecewiseTargetLoss(1e-6);
        // 4. Aufruf der Kompressionsfunktion
        AColGroup result = ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, in, cs);

        // 5. Ergebnis ist eine ColGroupPiecewiseLinearCompressed?
        assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);
        ColGroupPiecewiseLinearCompressed plGroup = (ColGroupPiecewiseLinearCompressed) result;

        // 6. Breakpoints per Getter, nicht per create()
        int[] breakpoints = plGroup.getBreakpoints();
        assertArrayEquals(new int[]{0, 20}, breakpoints);

        // 7. Pro Segment: 1 Segment → ein slope, ein intercept
        double[] slopes = plGroup.getSlopes();
        double[] intercepts = plGroup.getIntercepts();
        assertEquals(1, slopes.length);
        assertEquals(1, intercepts.length);

        // 8. Für konstante Daten: Steigung ~0, intercept ~1.0
        assertEquals(0.0, slopes[0], 1e-10);
        assertEquals(1.0, intercepts[0], 1e-10);

        // 9. Check: colIndexes stimmt
        IColIndex idx = plGroup.getColIndices();
        assertEquals(1, idx.size());
        assertEquals(0, idx.get(0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreate_nullBreakpoints() {
        int[] nullBp = null;
        ColGroupPiecewiseLinearCompressed.create(
                ColIndexFactory.create(new int[]{0}), nullBp, new double[]{1.0}, new double[]{0.0}, 10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreate_tooFewBreakpoints() {
        int[] singleBp = {0};
        ColGroupPiecewiseLinearCompressed.create(
                ColIndexFactory.create(new int[]{0}), singleBp, new double[]{1.0}, new double[]{0.0}, 10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreate_inconsistentSlopes() {
        int[] bp = {0, 5, 10};
        ColGroupPiecewiseLinearCompressed.create(
                ColIndexFactory.create(new int[]{0}), bp, new double[]{1.0, 2.0, 3.0},
                new double[]{0.0, 1.0}, 10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreate_inconsistentIntercepts() {
        int[] bp = {0, 5, 10};
        ColGroupPiecewiseLinearCompressed.create(
                ColIndexFactory.create(new int[]{0}), bp, new double[]{1.0, 2.0},
                new double[]{0.0}, 10);
    }

    @Test
    public void testCreate_validMultiSegment() {
        int[] bp = {0, 3, 7, 10};
        double[] slopes = {1.0, -2.0, 0.5};
        double[] intercepts = {0.0, 5.0, -1.0};
        IColIndex cols = ColIndexFactory.create(new int[]{0, 1});

        AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp, slopes, intercepts, 10);

        assertTrue(cg instanceof ColGroupPiecewiseLinearCompressed);
        assertNotSame(bp, ((ColGroupPiecewiseLinearCompressed) cg).getBreakpoints());
    }

    @Test
    public void testCreate_multiColumn() {
        IColIndex cols = ColIndexFactory.create(new int[]{5, 10, 15});
        int[] bp = {0, 5};
        double[] slopes = {3.0};
        double[] intercepts = {2.0};

        AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp, slopes, intercepts, 100);
        assertTrue(cg instanceof ColGroupPiecewiseLinearCompressed);
        assertTrue(cg instanceof ColGroupPiecewiseLinearCompressed);

        //
        assertTrue(cg.getNumValues() > 0);

        for (int r = 0; r < 5; r++) {
            double expected = 3.0 * r + 2.0;
            // colIdx=0 → globale Spalte 5
            assertEquals(expected, cg.getIdx(r, 0), 1e-9);
            // colIdx=1 → globale Spalte 10
            assertEquals(expected, cg.getIdx(r, 1), 1e-9);
            // colIdx=2 → globale Spalte 15
            assertEquals(expected, cg.getIdx(r, 2), 1e-9);
        }

        for (int r = 5; r < 10; r++) {
            double expected = 3.0 * r + 2.0;
            assertEquals(expected, cg.getIdx(r, 0), 1e-9);  // Alle Columns gleich
        }
        assertEquals(cols.size(), 3);
    }

    @Test
    public void testCreate_singleColumn() {
        IColIndex cols = ColIndexFactory.create(new int[]{5});
        int[] bp = {0, 5};
        double[] slopes = {3.0};
        double[] intercepts = {2.0};
        int numRows = 10;

        AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp, slopes, intercepts, numRows);

        assertTrue(cg instanceof ColGroupPiecewiseLinearCompressed);

        assertEquals(2.0, cg.getIdx(0, 0), 1e-9);  // 3*0 + 2
        assertEquals(5.0, cg.getIdx(1, 0), 1e-9);  // 3*1 + 2
    }

    @Test
    public void testCreate_validMinimal() {

        // 1 Segment: [0,10] → y = 2.0 * r + 1.0
        int[] bp = {0, 10};
        double[] slopes = {2.0};
        double[] intercepts = {1.0};
        IColIndex cols = ColIndexFactory.create(new int[]{0});
        int numRows = 10;

        AColGroup cg = ColGroupPiecewiseLinearCompressed.create(cols, bp, slopes, intercepts, numRows);

        // Korrekte Instanz
        assertTrue(cg instanceof ColGroupPiecewiseLinearCompressed);

        // getNumValues() > 0
        assertTrue(cg.getNumValues() > 0);

        // r < numRows
        for (int r = 0; r < numRows; r++) {
            double expected = 2.0 * r + 1.0;
            assertEquals("Row " + r, expected, cg.getIdx(r, 0), 1e-9);
        }

        // Letzte gültige Row
        assertEquals(19.0, cg.getIdx(9, 0), 1e-9);

        //Out-of-Bounds korrekt 0.0
        assertEquals(0.0, cg.getIdx(10, 0), 1e-9);
        assertEquals(0.0, cg.getIdx(9, 1), 1e-9);
    }

    @Test
    public void testDecompressToDenseBlock() {
        int[] bp = {0, 5, 10};
        double[] slopes = {1.0, 2.0};
        double[] intercepts = {0.0, 1.0};
        int numRows = 10;

        AColGroup cg = ColGroupPiecewiseLinearCompressed.create(
                ColIndexFactory.create(new int[]{0}), bp, slopes, intercepts, numRows);

        //  1. MatrixBlock mit korrekten Dimensionen
        MatrixBlock target = new MatrixBlock(numRows, 1, false);

        // 2. DenseBlock ZUERST alloziieren!
        target.allocateDenseBlock();  // Oder target.allocateDenseBlock(true);

        // 3. Jetzt DenseBlock verfügbar
        DenseBlock db = target.getDenseBlock();
        assertNotNull(db);  // Sicherstellen!

        // 4. Dekomprimieren
        cg.decompressToDenseBlock(db, 0, numRows, 0, 0);

        // 5. Prüfen
        for (int r = 0; r < numRows; r++) {
            double expected = (r < 5) ? 1.0 * r : 2.0 * r + 1.0;
            assertEquals("Row " + r, expected, db.get(r, 0), 1e-9);
        }
    }

    private ColGroupPiecewiseLinearCompressed createTestGroup(int numRows) {
        int[] bp = {0, 5, numRows};
        double[] slopes = {1.0, 3.0};
        double[] intercepts = {0.0, 2.0};
        return (ColGroupPiecewiseLinearCompressed) ColGroupPiecewiseLinearCompressed.create(
                ColIndexFactory.create(new int[]{0}), bp, slopes, intercepts, numRows);
    }

    @Test
    public void testDecompressToDenseBlock_fullRange() {
        ColGroupPiecewiseLinearCompressed cg = createTestGroup(12);

        MatrixBlock target = new MatrixBlock(12, 1, false);
        target.allocateDenseBlock();
        DenseBlock db = target.getDenseBlock();

        cg.decompressToDenseBlock(db, 0, 12, 0, 0);

        // Segment 0 [0,5): y = r
        assertEquals(0.0, db.get(0, 0), 1e-9);
        assertEquals(4.0, db.get(4, 0), 1e-9);

        assertEquals(17.0, db.get(5, 0), 1e-9);
        assertEquals(29.0, db.get(9, 0), 1e-9);
        assertEquals(32.0, db.get(10, 0), 1e-9);
        assertEquals(35.0, db.get(11, 0), 1e-9);
    }



    @Test
    public void testDecompressToDenseBlock_partialRange() {
        ColGroupPiecewiseLinearCompressed cg = createTestGroup(12);

        MatrixBlock target = new MatrixBlock(12, 1, false);
        target.allocateDenseBlock();
        DenseBlock db = target.getDenseBlock();

        // rl=6, ru=9 → r=6,7,8 dekomprimieren
        // offR=0 → schreibt in Target-Rows 6,7,8
        cg.decompressToDenseBlock(db, 6, 9, 0, 0);


        assertEquals(0.0, db.get(0, 0), 1e-9);   // Unberührt (vor rl=6)
        assertEquals(20.0, db.get(6, 0), 1e-9);
        assertEquals(23.0, db.get(7, 0), 1e-9);
        assertEquals(26.0, db.get(8, 0), 1e-9);
        assertEquals(0.0, db.get(9, 0), 1e-9);   // Unberührt (nach ru=9)
    }


    @Test
    public void testDecompressToDenseBlock_emptyRange() {
        ColGroupPiecewiseLinearCompressed cg = createTestGroup(12);

        MatrixBlock target = new MatrixBlock(5, 1, false);
        target.allocateDenseBlock();
        DenseBlock db = target.getDenseBlock();

        // Leerer Bereich
        cg.decompressToDenseBlock(db, 12, 12, 0, 0);  // rl=ru
        cg.decompressToDenseBlock(db, 3, 2, 0, 0);    // rl>ru

        // Alles bleibt 0.0
        for (int r = 0; r < 5; r++) {
            assertEquals(0.0, db.get(r, 0), 1e-9);
        }
    }

    @Test
    public void testDecompressToDenseBlock_nullSafety() {
        ColGroupPiecewiseLinearCompressed cg = createTestGroup(10);

        // Null DenseBlock
        cg.decompressToDenseBlock(null, 0, 10, 0, 0);

        // Ungültige Parameter (leerer Bereich)
        MatrixBlock target = new MatrixBlock(10, 1, false);
        target.allocateDenseBlock();
        DenseBlock db = target.getDenseBlock();

        cg.decompressToDenseBlock(db, 12, 12, 0, 0);  // rl == ru
        cg.decompressToDenseBlock(db, 5, 2, 0, 0);    // rl > ru

        // Target unverändert
        for (int r = 0; r < 10; r++) {
            assertEquals(0.0, db.get(r, 0), 1e-9);
        }
    }
    private CompressedSizeInfo createTestCompressedSizeInfo() {
        IColIndex cols = ColIndexFactory.create(new int[]{0});
        EstimationFactors facts = new EstimationFactors(2, 10);

        CompressedSizeInfoColGroup info = new CompressedSizeInfoColGroup(
                cols, facts, AColGroup.CompressionType.PiecewiseLinear);

        List<CompressedSizeInfoColGroup> infos = Arrays.asList(info);
        CompressedSizeInfo csi = new CompressedSizeInfo(infos);

        return csi;
    }

    @Test
    public void testCompressPiecewiseLinear_viaRealAPI() {

        MatrixBlock in = new MatrixBlock(10, 1, false);
        in.allocateDenseBlock();
        for (int r = 0; r < 10; r++) {
            in.set(r, 0, r * 0.5);
        }

        CompressionSettings cs = new CompressionSettingsBuilder()
                .addValidCompression(AColGroup.CompressionType.PiecewiseLinear)
                .create();

        CompressedSizeInfo csi = createTestCompressedSizeInfo();

        List<AColGroup> colGroups = ColGroupFactory.compressColGroups(in, csi, cs);

        boolean hasPiecewise = colGroups.stream()
                .anyMatch(cg -> cg instanceof ColGroupPiecewiseLinearCompressed);
        assertTrue(hasPiecewise);
    }

}