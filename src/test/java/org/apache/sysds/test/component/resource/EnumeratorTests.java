package org.apache.sysds.test.resource;

import org.apache.sysds.resource.enumeration.AnEnumerator;
import org.apache.sysds.resource.enumeration.EnumerationUtils;
import org.apache.sysds.runtime.controlprogram.Program;
import org.junit.Assert;
import org.junit.Test;

public class EnumeratorTests {

    @Test
    public void GridBasedEnumerationTest() {
        Program emptyProgram = new Program();

        AnEnumerator gridBasedEnumerator = (new AnEnumerator.Builder())
                .withRuntimeProgram(emptyProgram)
                .withEnumerationStrategy(AnEnumerator.EnumerationStrategy.GridBased)
                .withOptimizationStrategy(AnEnumerator.OptimizationStrategy.MinPrice)
                .withTimeLimit(Double.MAX_VALUE)
                .withNumberExecutorsRange(0, 2)
                .build();

        gridBasedEnumerator.setInstanceTable(TestingUtils.getSimpleCloudInstanceMap());

        gridBasedEnumerator.preprocessing();
        gridBasedEnumerator.processing();
        EnumerationUtils.SolutionPoint solution = gridBasedEnumerator.postprocessing();

        Assert.assertEquals("m5.xlarge", solution.driverInstance.getInstanceName());
        Assert.assertEquals(0, solution.numberExecutors);
    }

    @Test
    public void InterestBasedEnumerationTest() {
        Program emptyProgram = new Program();

        AnEnumerator gridBasedEnumerator = (new AnEnumerator.Builder())
                .withRuntimeProgram(emptyProgram)
                .withEnumerationStrategy(AnEnumerator.EnumerationStrategy.InterestBased)
                .withOptimizationStrategy(AnEnumerator.OptimizationStrategy.MinPrice)
                .withTimeLimit(Double.MAX_VALUE)
                .withNumberExecutorsRange(0, 2)
                .build();

        gridBasedEnumerator.setInstanceTable(TestingUtils.getSimpleCloudInstanceMap());

        gridBasedEnumerator.preprocessing();
        gridBasedEnumerator.processing();
        EnumerationUtils.SolutionPoint solution = gridBasedEnumerator.postprocessing();

        Assert.assertEquals("c5.xlarge", solution.driverInstance.getInstanceName());
        Assert.assertEquals(0, solution.numberExecutors);
    }

}
