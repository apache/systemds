/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.gpu.multigpu;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.AppenderSkeleton;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public abstract class GPUTest extends AutomatedTestBase {
    protected static final String TEST_DIR = "gpu/";
    protected static final String TEST_CLASS_DIR = TEST_DIR + MultiGPUTest.class.getSimpleName() + "/";
    protected static final String SINGLE_GPU_TEST = "SingleGPUTest";
    protected static final String MULTI_GPUS_TEST = "MultiGPUsTest";
    protected static final String TEST_NAME = "InferenceScript";
    protected static final String TRAIN_SCRIPT = "TrainScript";
    protected static final String DATA_SET = DATASET_DIR + "MNIST/mnist_test.csv";
    protected static final String SINGLE_TEST_CONFIG = CONFIG_DIR + "SystemDS-SingleGPU-config.xml";
    protected static final String MULTI_TEST_CONFIG = CONFIG_DIR + "SystemDS-config.xml";

    @Override
    public void setUp() {
        TEST_GPU = true;
        VERBOSE_STATS = true;
        addTestConfiguration(SINGLE_GPU_TEST,
                new TestConfiguration(TEST_CLASS_DIR, SINGLE_GPU_TEST, new String[] { "R" }));
        addTestConfiguration(MULTI_GPUS_TEST,
                new TestConfiguration(TEST_CLASS_DIR, MULTI_GPUS_TEST, new String[] { "R" }));
    }

    /**
     * Run the test with multiple GPUs
     *
     * @param multiGPUs whether to run the test with multiple GPUs
     */
    protected void runMultiGPUsTest(boolean multiGPUs, int numTestImages) {
        getAndLoadTestConfiguration(multiGPUs ? MULTI_GPUS_TEST : SINGLE_GPU_TEST);

        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TEST_NAME + ".dml";
        programArgs = new String[] { "-args", DATA_SET, output("R"), Integer.toString(numTestImages), "-config",
                multiGPUs ? MULTI_TEST_CONFIG : SINGLE_TEST_CONFIG };
        fullRScriptName = HOME + TEST_NAME + ".R";

        rCmd = null;
        InMemoryAppender appender = configureLog4j();

        runTest(true, false, null, -1);

        List<String> logs = appender.getLogMessages();
        int numRealThread = 0;
        for (String log : logs) {
            if (log.contains("has executed") && extractNumTasks(log) > 0) {
                numRealThread ++;
            }
        }
        if (multiGPUs) {
            assertTrue(numRealThread > 1);
        } else {
            assertEquals(1, numRealThread);
        }

        appender.clearLogMessages();
    }

    /**
     * Run the training script
     */
    protected void runTrainingScript(boolean multiGPUs, int numTestImages) {
        getAndLoadTestConfiguration(multiGPUs ? MULTI_GPUS_TEST : SINGLE_GPU_TEST);

        String HOME = SCRIPT_DIR + TEST_DIR;
        fullDMLScriptName = HOME + TRAIN_SCRIPT + ".dml";
        programArgs = new String[] { "-args", DATA_SET, output("R"), Integer.toString(numTestImages), "-config",
                multiGPUs ? MULTI_TEST_CONFIG : SINGLE_TEST_CONFIG };
        fullRScriptName = HOME + TEST_NAME + ".R";

        rCmd = null;
        InMemoryAppender appender = configureLog4j();

        runTest(true, false, null, -1);
    }

    protected static InMemoryAppender configureLog4j() {
        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.ERROR);

        Logger logger = Logger.getLogger(ParForProgramBlock.class.getName());
        logger.setLevel(Level.TRACE);

        InMemoryAppender inMemoryAppender = new InMemoryAppender();
        inMemoryAppender.setThreshold(Level.TRACE);
        logger.addAppender(inMemoryAppender);

        return inMemoryAppender;
    }

    protected static int extractNumTasks(String logMessage) {
        String regex = "has executed (\\d+) tasks";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(logMessage);
        if (matcher.find()) {
            return Integer.parseInt(matcher.group(1));
        }
        throw new IllegalArgumentException("No _numTasks value found in log message");
    }

    protected static class InMemoryAppender extends AppenderSkeleton {

        protected final List<String> logMessages = new ArrayList<>();

        @Override
        protected void append(LoggingEvent event) {
            if (event.getLevel().isGreaterOrEqual(Level.TRACE)) {
                logMessages.add(event.getRenderedMessage());
            }
        }

        @Override
        public void close() {
            // No resources to release
        }

        @Override
        public boolean requiresLayout() {
            return false;
        }

        public List<String> getLogMessages() {
            return new ArrayList<>(logMessages);
        }

        public void clearLogMessages() {
            logMessages.clear();
        }
    }
}
