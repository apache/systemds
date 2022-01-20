package org.apache.sysds.test.component.compress.dictionary;

import org.apache.sysds.runtime.compress.colgroup.dictionary.DeltaDictionary;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.junit.Assert;
import org.junit.Test;

public class DeltaDictionaryTest {

    @Test
    public void testScalarOpRightMultiplySingleColumn() {
        double scalar = 2;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2}, 1);
        ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {2, 4};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }

    @Test
    public void testScalarOpRightMultiplyTwoColumns() {
        double scalar = 2;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
        ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {2, 4, 6, 8};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }

    @Test
    public void testNegScalarOpRightMultiplyTwoColumns() {
        double scalar = -2;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
        ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {-2, -4, -6, -8};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }

    @Test
    public void testScalarOpLeftMultiplyTwoColumns() {
        double scalar = 2;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
        ScalarOperator sop = new LeftScalarOperator(Multiply.getMultiplyFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {2, 4, 6, 8};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }

    @Test
    public void testScalarOpRightDivideTwoColumns() {
        double scalar = 0.5;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
        ScalarOperator sop = new RightScalarOperator(Divide.getDivideFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {2, 4, 6, 8};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }

    @Test
    public void testScalarOpRightPlusSingleColumn() {
        double scalar = 2;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2}, 1);
        ScalarOperator sop = new RightScalarOperator(Plus.getPlusFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {3, 2};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }

    @Test
    public void testScalarOpRightPlusTwoColumns() {
        double scalar = 2;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
        ScalarOperator sop = new RightScalarOperator(Plus.getPlusFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {3, 4, 3, 4};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }

    @Test
    public void testScalarOpRightMinusTwoColumns() {
        double scalar = 2;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
        ScalarOperator sop = new RightScalarOperator(Minus.getMinusFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {-1, 0, 3, 4};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }

    @Test
    public void testScalarOpLeftPlusTwoColumns() {
        double scalar = 2;
        DeltaDictionary d = new DeltaDictionary(new double[] {1, 2, 3, 4}, 2);
        ScalarOperator sop = new LeftScalarOperator(Plus.getPlusFnObject(), scalar, 1);
        d = d.applyScalarOp(sop);
        double[] expected = new double[] {3, 4, 3, 4};
        Assert.assertArrayEquals(expected, d.getValues(), 0.01);
    }
}
