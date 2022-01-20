package org.apache.sysds.test.component.compress.colgroup;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;

@RunWith(value = Parameterized.class)
public class JolEstimateDeltaDDCTest extends JolEstimateTest {

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        ArrayList<Object[]> tests = new ArrayList<>();

        MatrixBlock mb;

        mb = DataConverter.convertToMatrixBlock(new double[][] {{0}});
        tests.add(new Object[] {mb});

        mb = DataConverter.convertToMatrixBlock(new double[][] {{1}});
        tests.add(new Object[] {mb});
        mb = DataConverter.convertToMatrixBlock(new double[][] {{1, 2, 3, 4, 5}});
        tests.add(new Object[] {mb});

        mb = DataConverter.convertToMatrixBlock(new double[][] {{1,2,3},{1,1,1}});
        tests.add(new Object[] {mb});

        mb = DataConverter
                .convertToMatrixBlock(new double[][] {{1,1},{2,1},{3,1},{4,1},{5,1}});
        tests.add(new Object[] {mb});

        mb = TestUtils.generateTestMatrixBlock(2, 5, 0, 20, 1.0, 7);
        tests.add(new Object[] {mb});

        return tests;
    }

    public JolEstimateDeltaDDCTest(MatrixBlock mb) {
        super(mb);
    }

    @Override
    public AColGroup.CompressionType getCT() {
        return delta;
    }
}
