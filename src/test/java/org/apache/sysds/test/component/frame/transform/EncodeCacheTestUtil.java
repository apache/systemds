package org.apache.sysds.test.component.frame.transform;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoder;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class EncodeCacheTestUtil {

    public static FrameBlock generateTestData(int numColumns, int numRows) {

        System.out.printf("Number of columns (number of runs to average over): %d%n", numColumns);
        System.out.printf("Number of rows (size of build result): %d%n", numRows);

        //create input data frame (numRows x numColumns)
        Types.ValueType[] valueTypes = new Types.ValueType[numColumns];
        for (int i = 0; i < numColumns; i++) {
            valueTypes[i] = Types.ValueType.FP32;
        }
        FrameBlock testData = TestUtils.generateRandomFrameBlock(numRows, valueTypes, 231);
        System.out.printf("Size of test data: %d x %d%n", testData.getNumColumns(), testData.getNumRows());

        return testData;
    }

    public static List<String> generateRecodeSpecs(int numColumns){
        List<String> recodeSpecs = new ArrayList<>();
        for (int i = 1; i <= numColumns; i++) {
            recodeSpecs.add("{recode:[C" + i + "]}");
        }
        return recodeSpecs;
    }

    public static List<String> generateBinSpecs(int numColumns){
        List<String> binSpecs = new ArrayList<>();
        for (int i = 1; i <= numColumns; i++) {
            binSpecs.add("{ids:true, bin:[{id:" + i + ", method:equi-width, numbins:4}]}");
        }
        return binSpecs;
    }

    static long measureBuildTime(ColumnEncoder encoder, FrameBlock data) {
        long startTime = System.nanoTime();
        encoder.build(data);
        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    static long measureEncodeTime(MultiColumnEncoder encoder, FrameBlock data, int k) {
        long startTime = System.nanoTime();
        encoder.encode(data, k);
        long endTime = System.nanoTime();
        return endTime - startTime;
    }

    static double analyzePerformance(int numExclusions, List<Long> runs, boolean withCache){
        //exclude the first x runs
        runs = runs.subList(numExclusions, runs.size());
        double avgExecTime = runs.stream()
                .collect(Collectors.averagingLong(Long::longValue));
        return avgExecTime/1_000_000.0;
    }

    static void compareMatrixBlocks(MatrixBlock mb1, MatrixBlock mb2) {
        assertEquals("Encoded matrix blocks should be equal", mb1, mb2);
    }
}
