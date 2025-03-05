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


package org.apache.sysds.test.component.misc;


import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.test.LoggingUtils;
import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.apache.sysds.api.DMLScript.executeScript;
import static org.apache.sysds.api.DMLScript.readDMLScript;

@net.jcip.annotations.NotThreadSafe
public class DMLScriptTest {

    @Test
    public void executeDMLScriptParsingExceptionTest() throws IOException {
        // Create a ListAppender to capture log messages
        final LoggingUtils.TestAppender appender = LoggingUtils.overwrite();
        try {
            Logger.getLogger(DMLScript.class).setLevel(Level.DEBUG);

            String[] args = new String[]{"-f", "test","-explain","XYZ"};
            Assert.assertFalse(executeScript(args));

            final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
            Assert.assertEquals(log.get(0).getMessage(), "Parsing Exception Invalid argument specified for -hops option, must be one of [hops, runtime, recompile_hops, recompile_runtime, codegen, codegen_recompile]");
        } finally {
            LoggingUtils.reinsert(appender);
        }
    }

    @Test
    public void executeDMLScriptAlreadySelectedExceptionTest() throws IOException {
        final LoggingUtils.TestAppender appender = LoggingUtils.overwrite();
        try {
            Logger.getLogger(DMLScript.class).setLevel(Level.DEBUG);

            String[] args = new String[]{"-f", "test", "-clean"};
            Assert.assertFalse(executeScript(args));

            final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
            Assert.assertEquals(log.get(0).getMessage(), "Mutually exclusive options were selected. The option 'clean' was specified but an option from this group has already been selected: 'f'");
        } finally {
            LoggingUtils.reinsert(appender);
        }
    }

    @Test
    public void executeDMLHelpTest() throws IOException {
        String[] args = new String[]{"-help"};
        Assert.assertTrue(executeScript(args));
    }

    @Test
    public void executeDMLCleanTest() throws IOException {
        String[] args = new String[]{"-clean"};
        Assert.assertTrue(executeScript(args));
    }

    @Test
    public void executeDMLfedMonitoringTest() {
        ExecutorService executor = Executors.newSingleThreadExecutor();

        try {
            String[] args = new String[]{"-fedMonitoring", "1"};
            Future<?> future = executor.submit(() -> executeScript(args));

            try {
                future.get(10, TimeUnit.SECONDS); // Wait for up to 10 seconds
            } catch (TimeoutException e) {
                future.cancel(true); // Cancel if timeout occurs
                System.out.println("Test fedMonitoring was forcefully terminated after 10s.");
            } catch (Exception e) {
                future.cancel(true); // Cancel in case of any other failure
                throw new RuntimeException("Test execution failed", e);
            }
        } finally {
            executor.shutdownNow();
        }
    }

    @Test(expected = RuntimeException.class)
    public void executeDMLfedMonitoringAddressTest1() throws Throwable {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            String[] args = new String[]{"-f","src/test/scripts/usertest/helloWorld.dml","-fedMonitoringAddress",
                    "http://localhost:8080"};
            Future<?> future = executor.submit(() -> executeScript(args));
            try {
                future.get(10, TimeUnit.SECONDS);
            } catch (TimeoutException e) {
                future.cancel(true);
                System.out.println("Test fedMonitoring was forcefully terminated after 10s.");
            } catch (Exception e) {
                future.cancel(true);
                throw e.getCause();
            }
        } finally {
            executor.shutdownNow();
            DMLScript.MONITORING_ADDRESS = null;
        }
    }

    @Test
    public void executeDMLfedMonitoringAddressTest2() throws Throwable {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            String[] args = new String[]{"-f","src/test/scripts/usertest/helloWorld.dml","-fedMonitoringAddress",
                    "https://example.com"};
            Future<?> future = executor.submit(() -> executeScript(args));
            try {
                future.get(10, TimeUnit.SECONDS);
            } catch (TimeoutException e) {
                future.cancel(true);
                System.out.println("Test fedMonitoring was forcefully terminated after 10s.");
            } catch (Exception e) {
                future.cancel(true);
                throw e.getCause();
            }
        } finally {
            executor.shutdownNow();
            DMLScript.MONITORING_ADDRESS = null;
        }
    }

    @Test
    public void executeDMLWithScriptTest() throws IOException {
        String cl = "systemds -s \"print('hello')\"";
        String[] args = cl.split(" ");
        final PrintStream originalOut = System.out;
        final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();

        System.setOut(new PrintStream(outputStreamCaptor));
        try{
            Assert.assertTrue(executeScript(args));
            Assert.assertEquals("hello", outputStreamCaptor.toString().split(System.lineSeparator())[0]);
        } finally {
            System.setOut(originalOut);
        }
    }

    @Test(expected = LanguageException.class)
    public void readDMLWithNoScriptTest() throws IOException {
        readDMLScript(false, null);
    }

    @Test(expected = LanguageException.class)
    public void readDMLWithNoFilepathTest() throws IOException {
        readDMLScript(true, null);
    }

    @Test(expected = IOException.class)
    public void readDMLWrongHDFSPathTest1() throws IOException {
        readDMLScript(true, "hdfs:/namenodehost/test.txt");
    }

    @Test(expected = IllegalArgumentException.class)
    public void readDMLWrongHDFSPathTes2t() throws IOException {
        readDMLScript(true, "hdfs://namenodehost/test.txt");
    }

    @Test(expected = IOException.class)
    public void readDMLWrongGPFSPathTest() throws IOException {
        readDMLScript(true, "gpfs:/namenodehost/test.txt");
    }

    @Test
    public void setActiveAMTest(){
        DMLScript.setActiveAM();
        try {

            Assert.assertTrue(DMLScript.isActiveAM());
        } finally {
            DMLScript._activeAM = false;
        }
    }

    @Test
    public void runDMLScriptMainLanguageExceptionTest(){
        String cl = "systemds -debug -s \"printx('hello')\"";
        String[] args = cl.split(" ");
        final PrintStream originalErr = System.err;

        try {
            final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();
            System.setErr(new PrintStream(outputStreamCaptor));
            DMLScript.main(args);
            System.setErr(originalErr);
            Assert.assertTrue(outputStreamCaptor.toString().split(System.lineSeparator())[0]
                    .startsWith("org.apache.sysds.parser.LanguageException: ERROR: [line 1:0] -> printx('hello') -- function printx is undefined"));
        } finally {
            System.setErr(originalErr);
        }

    }

    @Test
    public void runDMLScriptMainDMLRuntimeExceptionTest(){
        String cl = "systemds -s \"F=as.frame(matrix(1,1,1));spec=\"{ids:true,recod:[1]}\";" +
                "M=transformapply(target=F,spec=spec,meta=F);print(M[1,1]) \"";
        String[] args = cl.split(" ");

        final PrintStream originalOut = System.out;
        try {
            final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();
            System.setOut(new PrintStream(outputStreamCaptor));
            DMLScript.main(args);
            System.setOut(originalOut);
            String[] lines = outputStreamCaptor.toString().split(System.lineSeparator());
            for (int i = 0; i < lines.length; i++) {
                if(lines[i].startsWith("An Error Occurred :")){
                    for (int j = 0; j < 4; j++) {
                        Assert.assertTrue(lines[i + 1 + j].trim().startsWith("DMLRuntimeException"));
                    }
                    break;
                }
            }
        } finally {
            System.setOut(originalOut);
        }

    }

    @Test(expected = RuntimeException.class)
    public void executeDMLWithScriptInvalidConfTest1() throws IOException {
        String cl = "systemds -config src/test/resources/conf/invalid-gpu-conf.xml -s \"print('hello')\"";
        String[] args = cl.split(" ");
        executeScript(args);
    }

    @Test(expected = RuntimeException.class)
    public void executeDMLWithScriptInvalidConfTest2() throws IOException {
        String cl = "systemds -config src/test/resources/conf/invalid-shadow-buffer1-conf.xml -s \"print('hello')\"";
        String[] args = cl.split(" ");
        executeScript(args);
    }

    @Test(expected = RuntimeException.class)
    public void executeDMLWithScriptInvalidConfTest3() throws IOException {
        String cl = "systemds -config src/test/resources/conf/invalid-shadow-buffer2-conf.xml -s \"print('hello')\"";
        String[] args = cl.split(" ");
        executeScript(args);
    }

    @Test
    public void executeDMLWithScriptValidCodegenConfTest() throws IOException {
        String cl = "systemds -config src/test/resources/conf/invalid-codegen-conf.xml -s \"print('hello')\"";
        String[] args = cl.split(" ");
        executeScript(args);
    }

    @Test
    public void executeDMLWithScriptShadowBufferWarnTest() throws IOException {
        String cl = "systemds -config src/test/resources/conf/shadow-buffer-conf.xml -s \"print('hello')\"";
        String[] args = cl.split(" ");
        DMLScript.EVICTION_SHADOW_BUFFER_CURR_BYTES =1000000000L;

        final PrintStream originalOut = System.out;
        try {
            final ByteArrayOutputStream outputStreamCaptor = new ByteArrayOutputStream();
            System.setOut(new PrintStream(outputStreamCaptor));
            executeScript(args);
            System.setOut(originalOut);
            String[] lines = outputStreamCaptor.toString().split(System.lineSeparator());
            Assert.assertTrue(lines[0].startsWith("WARN: Cannot use the shadow buffer due to potentially cached GPU objects. Current shadow buffer size (in bytes)"));
        } finally {
            System.setOut(originalOut);
        }
    }

    @Test
    public void executeDMLWithScriptAndInfoTest() throws IOException {
        String cl = "systemds -s \"print('hello')\"";
        String[] args = cl.split(" ");
        Logger.getLogger(DMLScript.class).setLevel(Level.INFO);
        final LoggingUtils.TestAppender appender = LoggingUtils.overwrite();
        try {
            Assert.assertTrue(executeScript(args));
            final List<LoggingEvent> log = LoggingUtils.reinsert(appender);
            try {
                int i = log.get(0).getMessage().toString().startsWith("Low memory budget") ? 1 : 0;
                Assert.assertTrue(log.get(i++).getMessage().toString().startsWith("BEGIN DML run"));
                Assert.assertTrue(log.get(i).getMessage().toString().startsWith("Process id"));
            } catch (Error e) {
                System.out.println("ERROR while evaluating INFO logs: ");
                for (LoggingEvent loggingEvent : log) {
                    System.out.println(loggingEvent.getMessage());
                }
                throw e;
            }

        } finally {
            LoggingUtils.reinsert(appender);
        }
    }

    @Test
    public void executeDMLWithScriptAndDebugTest() throws IOException {
        // have to run sequentially, to avoid concurrent call to Logger.getLogger(DMLScript.class)
        String cl = "systemds -s \"print('hello')\"";
        String[] args = cl.split(" ");

        Logger.getLogger(DMLScript.class).setLevel(Level.DEBUG);
        final LoggingUtils.TestAppender appender2 = LoggingUtils.overwrite();
        try{
            Assert.assertTrue(executeScript(args));
            final List<LoggingEvent> log = LoggingUtils.reinsert(appender2);
            try {
                int i = log.get(0).getMessage().toString().startsWith("Low memory budget") ? 2 : 1;
                Assert.assertTrue(log.get(i++).getMessage().toString().startsWith("BEGIN DML run"));
                Assert.assertTrue(log.get(i++).getMessage().toString().startsWith("DML script"));
                Assert.assertTrue(log.get(i).getMessage().toString().startsWith("Process id"));
            } catch (Error e){
                for (LoggingEvent loggingEvent : log) {
                    System.out.println(loggingEvent.getMessage());
                }
                throw e;
            }
        } finally {
            LoggingUtils.reinsert(appender2);
        }
    }

    @Test
    public void createDMLScriptInstance(){
        DMLScript script = new DMLScript();
        Assert.assertTrue(script != null);
    }
}
