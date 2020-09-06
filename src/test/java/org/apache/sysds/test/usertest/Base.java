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

package org.apache.sysds.test.usertest;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;

public class Base {
    public final static String BASE_FOLDER = "src/test/scripts/usertest/";

    /**
     * Run the system in a different JVM
     * 
     * @param script a path to the script to execute.
     * @return A pair of standard out and standard error.
     */
    public static Pair<String, String> runProcess(String script) {
        String fullDMLScriptName = BASE_FOLDER + script;

        String separator = System.getProperty("file.separator");
        String classpath = System.getProperty("java.class.path");
        String path = System.getProperty("java.home") + separator + "bin" + separator + "java";
        ProcessBuilder processBuilder = new ProcessBuilder(path, "-cp", classpath, DMLScript.class.getName(), "-f",
            fullDMLScriptName);

        StringBuilder stdout = new StringBuilder();
        StringBuilder stderr = new StringBuilder();
        try {
            Process process = processBuilder.start();

            BufferedReader output = new BufferedReader(new InputStreamReader(process.getInputStream()));
            BufferedReader error = new BufferedReader(new InputStreamReader(process.getErrorStream()));

            Thread t = new Thread(() -> {
                output.lines().forEach(s -> stdout.append("\n" + s));
            });

            Thread te = new Thread(() -> {
                error.lines().forEach(s -> stderr.append("\n" + s));
            });

            t.start();
            te.start();

            process.waitFor();
            t.join();
            te.join();
        }
        catch(IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return new ImmutablePair<>(stdout.toString(), stderr.toString());
    }

    public static Pair<String, String> runThread(String script) {
        String fullDMLScriptName = BASE_FOLDER + script;
        return runThread(new String[]{"-f", fullDMLScriptName});
    }

    public static Pair<String, String> runThread(String[] args) {
        Thread t = new Thread(() -> {
            DMLScript.main(args);
        });
        
        ByteArrayOutputStream buff = new ByteArrayOutputStream();
        ByteArrayOutputStream buffErr = new ByteArrayOutputStream();
        PrintStream old = System.out;
        PrintStream oldErr = System.err;
        System.setOut(new PrintStream(buff));
        System.setErr(new PrintStream(buffErr));
        
        t.start();
        try {
            t.join();
        }
        catch(InterruptedException e) {
            e.printStackTrace();
        }

        System.setOut(old);
        System.setErr(oldErr);
        
        return new ImmutablePair<>(buff.toString(), buffErr.toString());
    }
}
