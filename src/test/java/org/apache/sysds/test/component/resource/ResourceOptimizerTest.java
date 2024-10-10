package org.apache.sysds.test.component.resource;

import org.apache.commons.cli.ParseException;
import org.apache.sysds.resource.ResourceOptimizer;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.apache.sysds.test.component.resource.TestingUtils.TEST_REGION;

public class ResourceOptimizerTest {

    @Test
    public void executeMainForL2SVMTest() throws IOException, ParseException {
        File tmpRegionFile = TestingUtils.generateTmpFeeTableFile();
        File tmpInfoFile = TestingUtils.generateTmpInstanceInfoTableFile();
        Path tempOutFolder = Files.createTempDirectory("out");

        // TODO: fix why for "-nvargs", "m=10000000", "n=100000" time and price is lower than for "-nvargs", "m=1000000", "n=100000"
        String[] args = {
                "-f", "src/test/scripts/component/resource/Algorithm_L2SVM.dml",
                "-infoTable", tmpInfoFile.getPath(),
                "-region", TEST_REGION,
                "-regionTable", tmpRegionFile.getPath(),
                "-output", tempOutFolder.toString(),
                "-maxExecutors", "20",
                "-nvargs", "m=1000000", "n=10000"
        };
        ResourceOptimizer.main(args);

        Files.deleteIfExists(tmpRegionFile.toPath());
        Files.deleteIfExists(tmpInfoFile.toPath());
//        Files.deleteIfExists(tempOutFolder);
    }
}
