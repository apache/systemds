package org.apache.sysds.runtime.compress.colgroup;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.indexes.ArrayIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ColGroupPiecewiseLinearCompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.Arrays;
import java.util.List;

import static org.apache.sysds.runtime.compress.colgroup.ColGroupFactory.*;
import static org.junit.Assert.*;

/**
 * Tests für PiecewiseLinearColGroupCompressed, fokussiert auf:
 * - Konstruktor / create(...)
 * - decompressToDenseBlock(...)
 */
//TODO Fix
public class ColGroupPiecewiseLinearCompressedTest {

    private CompressionSettings cs;
    // -------------------------------------------------------------
    // 1. create(...) und Konstruktor
    // -------------------------------------------------------------

    @BeforeEach
    void setUp() {
        CompressionSettings cs = new CompressionSettingsBuilder().create();

    }

    @Test
    public void testComputeBreakpoints_uniformColumn() {
        //cs.setPiecewiseTargetLoss(1e-3);
        double[] column = {1.0, 1.0, 1.0, 1.0, 1.0}; // ← Test-spezifisch
        List<Integer> breaks = computeBreakpoints(cs, column,1e-3);
        assertEquals(Arrays.asList(0,5), breaks); // Erwartet: keine Breaks
    }

    @Test
    public void testComputeBreakpoints_linearIncreasing() {
        //cs.setPiecewiseTargetLoss(1e-3);
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0}; // ← andere column
        List<Integer> breaks = computeBreakpoints(cs, column,1e-3);
        assertEquals(Arrays.asList(0, 5), breaks); // Erwartet

    }

    @Test
    public void testComputeBreakpoints_highLoss_uniform() {
        //cs.setPiecewiseTargetLoss(1.0); // ← andere Loss
        double[] column = {1.0, 1.0, 1.0, 1.0, 1.0};
        List<Integer> breaks = computeBreakpoints(cs, column,10000.0);
        assertEquals(Arrays.asList(0,5), breaks);
    }
    @Test
    public void testComputeBreakpoints_twoSegments() {
        // {1,1,1, 2,2,2} → 2 Segmente → [0,3,6]
        double[] column = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};
        var breaks = computeBreakpoints(cs, column, 1e-3);
        assertEquals(Arrays.asList(0, 3, 6), breaks);
    }

    @Test
    public void testComputeBreakpoints_noLoss_linear() {
        //cs.setPiecewiseTargetLoss(0.0);
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0};
        List<Integer> breaks = computeBreakpoints(cs, column,0.0);
        assertEquals(Arrays.asList(0,5), breaks); // bei 0 Loss alle Breaks
    }
    @Test
    public void testComputeBreakpointsLambda_const() {
        double[] column = {1.0, 1.0, 1.0, 1.0, 1.0};  // 5 Werte
        List<Integer> breaks = computeBreakpointsLambda(column, 5.0);
        assertEquals(Arrays.asList(0, 5), breaks);  // 0 bis 5

        breaks = computeBreakpointsLambda(column, 0.01);
        assertEquals(Arrays.asList(0, 5), breaks);  // auch mit kleinem lambda
    }
    @Test
    public void testComputeBreakpointsLambda_twoSegments() {
        double[] column = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0};  // 6 Werte

        // mit kleinem lambda -> viele Segmente (kostenlos fast)
        List<Integer> breaks = computeBreakpointsLambda(column, 0.01);
        assertTrue(breaks.contains(3));  // 3 muss als Grenze enthalten sein
        assertEquals(3, breaks.size()); // 0, 3, 6
        assertEquals(Arrays.asList(0, 3, 6), breaks);

        // mit großem lambda -> nur ein Segment
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
        double[] column = {0.0, 2.0, 0.0, 4.0, 0.0, 6.0};  // 6 Punkte

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
        List<Integer> breaks = Arrays.asList(2, 5, 8); // Segmente [2,5), [5,8)

        double total = computeTotalSSE(column, breaks);
        double sse1 = computeSegmentCost(column, 2, 5);
        double sse2 = computeSegmentCost(column, 5, 8);

        // Resultat: Summe der zwei Segmente
        assertEquals(sse1 + sse2, total, 1e-10);

        // Die Indizes <2 und >=8 sind nicht Teil der Segmente und fließen nicht in totalSSE ein
    }
    @Test
    public void testComputeTotalSSE_oneSegment_identical() {
        double[] column = {1.0, 2.0, 3.0, 4.0, 5.0};

        // Vergleich: SSE des gesamten Segments über [0,5)
        double sseTotal = computeSegmentCost(column, 0, 5);

        // Berechnung mit computeTotalSSE und breaks [0,5]
        List<Integer> breaks = Arrays.asList(0, 5);
        double total = computeTotalSSE(column, breaks);

        // beide müssen exakt gleich sein
        assertEquals(sseTotal, total, 1e-10);
    }
    @Test
    public void testComputeTotalSSE_nonConstant() {
        double[] column = {0.0, 1.0, 2.0, 3.0, 4.0};
        List<Integer> breaks = Arrays.asList(0, 2, 5); // [0,2), [2,5)

        double total = computeTotalSSE(column, breaks);
        double sse1 = computeSegmentCost(column, 0, 2);
        double sse2 = computeSegmentCost(column, 2, 5);

        // Sanity-Check: Ergebnis positiv, Summe der beiden SSE
        assertTrue(total >= 0.0);
        assertEquals(sse1 + sse2, total, 1e-10);
    }
    @Test
    public void testComputeTotalSSE_edgeCases() {
        // Leere Spalte, Segmente [0,0] → kein Segment
        double[] columnEmpty = {}; // length 0
        List<Integer> breaksEmpty = Arrays.asList(0, 0);
        assertEquals(0.0, computeTotalSSE(columnEmpty, breaksEmpty), 1e-10);

        // Spalte der Länge 1, ein Segment [0,1)
        double[] columnOne = {42.0};
        List<Integer> breaksOne = Arrays.asList(0, 1);
        double total = computeTotalSSE(columnOne, breaksOne);
        assertEquals(0.0, total, 1e-10);
    }
    @Test
    public void testRegressSegment_empty() {
        double[] column = {1.0, 2.0, 3.0};
        double[] result = regressSegment(column, 0, 0); // leer
        assertEquals(0.0, result[0], 1e-10);
        assertEquals(0.0, result[1], 1e-10);
    }
    @Test
    public void testRegressSegment_singlePoint() {
        double[] column = {1.0, 2.0, 3.0};
        double[] result = regressSegment(column, 1, 2); // nur i=1: y=2.0

        assertEquals(0.0, result[0], 1e-10);          // slope = 0
        assertEquals(2.0, result[1], 1e-10);          // intercept = Mittelwert
    }
    @Test
    public void testRegressSegment_twoIdentical() {
        double[] column = {5.0, 5.0, 1.0, 1.0};
        double[] result = regressSegment(column, 0, 2); // i=0:5, i=1:5

        // Steigung = 0, y = 5.0 + 0*i
        assertEquals(0.0, result[0], 1e-10);
        assertEquals(5.0, result[1], 1e-10);
    }
    @Test
    public void testRegressSegment_twoPoints() {
        double[] column = {0.0, 2.0}; // (i=0, y=0), (i=1, y=2)
        double[] result = regressSegment(column, 0, 2);

        // Gerade durch (0,0) und (1,2) → y = 2*i + 0
        assertEquals(2.0, result[0], 1e-10);
        assertEquals(0.0, result[1], 1e-10);
    }
    @Test
    public void testRegressSegment_twoPoints_offset() {
        // column[0], column[1], column[2], column[3] → es gibt 4 Werte
        double[] column = {1.0, 3.0, 5.0, 7.0};  // z. B. y = 2*x + 1 → bei x=2: y=5, x=3: y=7
        double[] result = regressSegment(column, 2, 4);  // Segment [2,4) → i=2,3

        // Gerade durch (2,5), (3,7): slope = 2, intercept = 1
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

        // Exakt: slope = 1.5, intercept = 2.0
        assertEquals(a, result[0], 1e-10);
        assertEquals(b, result[1], 1e-10);
    }
    @Test
    public void testRegressSegment_denomZero() {
        // fiktiv: ein Segment mit einem Punkt
        double[] column = {10.0};
        double[] result = regressSegment(column, 0, 1);

        assertEquals(0.0, result[0], 1e-10);
        assertEquals(10.0, result[1], 1e-10);
    }

    @Test
    public void testCompressPiecewiseLinearFunctional_const() {
        // 1. MatrixBlock mit einer konstanten Spalte erzeugen
        double[] data = {1.0, 1.0, 1.0, 1.0, 1.0};  // 5 Zeilen, 1 Spalte
        MatrixBlock in = new MatrixBlock(5, 1, false).quickSetMatrix(data, 5);

        // 2. colIndexes für Spalte 0
        IColIndex colIndexes = ColIndexFactory.create(0);

        // 3. Aufruf der Kompressionsfunktion
        AColGroup result = ColGroupFactory.compressPiecewiseLinearFunctional(colIndexes, in, new CompressionSettings());

        // 4. Ergebnis ist eine ColGroupPiecewiseLinearCompressed?
        assertTrue(result instanceof ColGroupPiecewiseLinearCompressed);
        ColGroupPiecewiseLinearCompressed plGroup = (ColGroupPiecewiseLinearCompressed) result;

        // 5. Check Breakpoints: [0, 5] → ein Segment
        int[] breakpoints = plGroup.c();
        assertArrayEquals(new int[] {0, 5}, breakpoints);

        // 6. Pro Segment: 1 Segment → ein slope, ein intercept
        double[] slopes = plGroup.getSlopes();
        double[] intercepts = plGroup.getIntercepts();
        assertEquals(1, slopes.length);
        assertEquals(1, intercepts.length);

        // 7. Für konstante Daten: Steigung ~0, intercept ~1.0
        assertEquals(0.0, slopes[0], 1e-10);
        assertEquals(1.0, intercepts[0], 1e-10);  // Mittelwert der Spalte

        // 8. Check: colIndexes stimmt
        assertEquals(1, plGroup.getColIndex().size());
        assertEquals(0, plGroup.getColIndex().get(0));
    }




}