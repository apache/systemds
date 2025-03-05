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


package org.apache.sysds.test.api;

import org.apache.commons.cli.ParseException;
import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.runtime.lineage.LineageCacheConfig;
import org.junit.Assert;
import org.junit.Test;

import javax.validation.constraints.AssertTrue;

import static org.apache.sysds.api.DMLOptions.parseCLArguments;

public class DMLOptionsTest {

    @Test
    public void parseCLArgumentsLineageTest() throws ParseException {
        String[] args = new String[]{"-f", "test", "-lineage", "policy_dagheight"};
        DMLOptions opts = parseCLArguments(args);
        Assert.assertTrue(opts.lineage && opts.linCachePolicy == LineageCacheConfig.LineageCachePolicy.DAGHEIGHT);

        args = new String[]{"-f", "test", "-lineage", "estimate"};
        opts = parseCLArguments(args);
        Assert.assertTrue(opts.lineage && opts.lineage_estimate);
    }

    @Test
    public void parseCLArgumentsGPUExplainTest() throws ParseException {
        String[] args = new String[]{"-f", "test", "-gpu",};
        parseCLArguments(args);
        args = new String[]{"-f", "test","-explain","XYZ"};
        try {
            parseCLArguments(args);
        } catch (ParseException e) {
            assert e.getMessage().equals("Invalid argument specified for -hops option, must be one of [hops, runtime, recompile_hops, recompile_runtime, codegen, codegen_recompile]");
        }
    }

    @Test
    public void parseCLArgumentsNGramsTest() throws ParseException {
        String[] args = new String[]{"-f", "test", "-ngrams",};
        parseCLArguments(args);
        args = new String[]{"-f", "test", "-ngrams","1"};
        parseCLArguments(args);
        args = new String[]{"-f", "test", "-ngrams","1","1","FALSE"};
        parseCLArguments(args);
        args = new String[]{"-f", "test","-ngrams","1,2","b"};
        try {
            parseCLArguments(args);
        } catch (ParseException e) {
            assert e.getMessage().equals("Invalid argument specified for -ngrams option, must be a valid integer");
        }
    }

    @Test
    public void parseCLArgumentsFEDStatsTest() throws ParseException {
        String[] args = new String[]{"-f", "test", "-fedStats",};
        parseCLArguments(args);

        args = new String[]{"-f", "test", "-fedStats", "21"};
        DMLOptions opts = parseCLArguments(args);
        Assert.assertEquals(21, opts.fedStatsCount);

        args = new String[]{"-f", "test", "-fedStats", "xyz"};
        try {
            parseCLArguments(args);
        } catch (ParseException e) {
            assert e.getMessage().equals("Invalid argument specified for -fedStats option, must be a valid integer");
        }
    }

    @Test
    public void parseCLArgumentsFEDMonitoringTest() throws ParseException {
        String[] args = new String[]{"-fedMonitoring"};
        try {
            parseCLArguments(args);
        } catch (ParseException e) {
            assert e.getMessage().equals("No port [integer] specified for -fedMonitoring option");
        }

        args = new String[]{"-fedMonitoring","21", "-fedMonitoringAddress"};
        try {
            parseCLArguments(args);
        } catch (ParseException e) {
            assert e.getMessage().equals("No address [String] specified for -fedMonitoringAddress option");
        }

        args = new String[]{"-fedMonitoring", "21"};
        DMLOptions opts = parseCLArguments(args);
        Assert.assertTrue(opts.fedMonitoring);
        Assert.assertEquals(21, opts.fedMonitoringPort);

        args = new String[]{"-fedMonitoring", "21", "-fedMonitoringAddress", "xyz"};
        opts = parseCLArguments(args);
        Assert.assertTrue(opts.fedMonitoring);
        Assert.assertEquals(21, opts.fedMonitoringPort);
        Assert.assertEquals("xyz", opts.fedMonitoringAddress);
    }

    @Test
    public void parseCLArgumentsFEDCompilationTest() throws ParseException {
        String[] args = new String[]{"-f", "test", "-federatedCompilation"};
        parseCLArguments(args);

        args = new String[]{"-f", "test", "-federatedCompilation", "1=NONE"};
        DMLOptions opts = parseCLArguments(args);
        Assert.assertTrue(opts.federatedCompilation);
    }
    @Test
    public void parseCLArgumentsFEDNoRuntimeConversonTest() throws ParseException {
        String[] args = new String[]{"-f", "test", "-noFedRuntimeConversion"};
        DMLOptions opts = parseCLArguments(args);
        Assert.assertTrue(opts.noFedRuntimeConversion);
    }

}
