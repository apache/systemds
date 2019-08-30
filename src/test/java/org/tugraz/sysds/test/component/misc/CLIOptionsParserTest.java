/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.test.component.misc;

import java.util.Map;

import org.apache.commons.cli.AlreadySelectedException;
import org.apache.commons.cli.MissingOptionException;
import org.apache.commons.cli.ParseException;
import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLOptions;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.utils.Explain;


public class CLIOptionsParserTest {

	@Test(expected = MissingOptionException.class)
	public void testNoOptions() throws Exception {
		String cl = "systemds";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test
	public void testFile() throws Exception {
		String cl = "systemds -f test.dml";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals("test.dml", o.filePath);
	}

	@Test
	public void testScript() throws Exception {
		String cl = "systemds -s \"print('hello')\"";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals("print('hello')", o.script);
	}

	@Test
	public void testConfig() throws Exception {
		String cl = "systemds -s \"print('hello')\" -config SystemDS-config.xml";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals("print('hello')", o.script);
		Assert.assertEquals("SystemDS-config.xml", o.configFile);
	}

	@Test
	public void testDebug() throws Exception {
		String cl = "systemds -s \"print('hello')\" -debug";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals("print('hello')", o.script);
		Assert.assertEquals(true, o.debug);
	}

	@Test
	public void testClean() throws Exception {
		String cl = "systemds -clean";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.clean);
	}

	@Test(expected = AlreadySelectedException.class)
	public void testBadClean() throws Exception {
		String cl = "systemds -clean -f test.dml";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test(expected = AlreadySelectedException.class)
	public void testBadScript() throws Exception {
		String cl = "systemds -f test.dml -s \"print('hello')\"";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test
	public void testStats() throws Exception {
		String cl = "systemds -f test.dml -stats";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.stats);
		Assert.assertEquals(10, o.statsCount);
	}

	@Test
	public void testStatsCount() throws Exception {
		String cl = "systemds -f test.dml -stats 9123";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.stats);
		Assert.assertEquals(9123, o.statsCount);
	}

	@Test(expected = ParseException.class)
	public void testBadStats() throws Exception {
		String cl = "systemds -f test.dml -stats help";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.stats);
	}

    @Test
    public void testLineage() throws Exception {
        String cl = "systemds -f test.dml -lineage";
        String[] args = cl.split(" ");
        DMLOptions o = DMLOptions.parseCLArguments(args);
        Assert.assertEquals(true, o.lineage);
		Assert.assertEquals(false, o.lineage_dedup);
		Assert.assertEquals(false, o.lineage_reuse);
    }
    
	@Test
	public void testLineageDedup() throws Exception {
		String cl = "systemds -f test.dml -lineage dedup";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.lineage);
		Assert.assertEquals(true, o.lineage_dedup);
		Assert.assertEquals(false, o.lineage_reuse);
	}
	
	@Test
	public void testLineageReuse() throws Exception {
		String cl = "systemds -f test.dml -lineage reuse";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.lineage);
		Assert.assertEquals(true, o.lineage_reuse);
		Assert.assertEquals(false, o.lineage_dedup);
	}
	
	@Test
	public void testLineageDedupAndReuse() throws Exception {
		String cl = "systemds -f test.dml -lineage dedup reuse";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.lineage);
		Assert.assertEquals(true, o.lineage_dedup);
		Assert.assertEquals(true, o.lineage_reuse);
	}
	
	@Test(expected = ParseException.class)
	public void testBadLineageOptionDedup() throws Exception {
		String cl = "systemds -f test.dml -lineage ded";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}
	
	@Test(expected = ParseException.class)
	public void testBadLineageOptionReuse() throws Exception {
		String cl = "systemds -f test.dml -lineage rese";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}
	
	@Test
	public void testGPUForce() throws Exception {
		String cl = "systemds -f test.dml -gpu force";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.gpu);
		Assert.assertEquals(true, o.forceGPU);
	}

	@Test(expected = ParseException.class)
	public void testBadGPUOption() throws Exception {
		String cl = "systemds -f test.dml -gpu f2orce";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test
	public void testHelp() throws Exception {
		String cl = "systemds -help";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.help);
	}

	@Test(expected = AlreadySelectedException.class)
	public void testBadHelp() throws Exception {
		String cl = "systemds -help -clean";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(true, o.help);
	}

	@Test
	public void testExplain1() throws Exception {
		String cl = "systemds -f test.dml -explain";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(Explain.ExplainType.RUNTIME, o.explainType);
	}

	@Test
	public void testExplain2() throws Exception {
		String cl = "systemds -f test.dml -explain hops";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(Explain.ExplainType.HOPS, o.explainType);
	}

	@Test
	public void testExplain3() throws Exception {
		String cl = "systemds -f test.dml -explain runtime";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(Explain.ExplainType.RUNTIME, o.explainType);
	}

	@Test
	public void testExplain4() throws Exception {
		String cl = "systemds -f test.dml -explain recompile_hops";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(Explain.ExplainType.RECOMPILE_HOPS, o.explainType);
	}

	@Test
	public void testExplain5() throws Exception {
		String cl = "systemds -f test.dml -explain recompile_runtime";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(Explain.ExplainType.RECOMPILE_RUNTIME, o.explainType);
	}

	@Test
	public void testExec2() throws Exception {
		String cl = "systemds -f test.dml -exec spark";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(ExecMode.SPARK, o.execMode);
	}

	@Test
	public void testExec3() throws Exception {
		String cl = "systemds -f test.dml -exec singlenode";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(ExecMode.SINGLE_NODE, o.execMode);
	}

	@Test
	public void testExec4() throws Exception {
		String cl = "systemds -f test.dml -exec hybrid";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Assert.assertEquals(ExecMode.HYBRID, o.execMode);
	}

	@Test(expected = ParseException.class)
	public void testBadExec() throws Exception {
		String cl = "systemds -f test.dml -exec new_system";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test
	public void testArgs1() throws Exception {
		String cl = "systemds -f test.dml -args 10 \"x.csv\"";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Map<String, String> m = o.argVals;
		Assert.assertEquals(2, m.size());
		Assert.assertEquals("10", m.get("$1"));
		Assert.assertEquals("x.csv", m.get("$2"));
	}

	@Test
	public void testArgs2() throws Exception {
		String cl = "systemds -f test.dml -args 10 \"x.csv\" 1234.2 systemds.conf -config systemds.conf";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Map<String, String> m = o.argVals;
		Assert.assertEquals(4, m.size());
		Assert.assertEquals("10", m.get("$1"));
		Assert.assertEquals("x.csv", m.get("$2"));
		Assert.assertEquals("1234.2", m.get("$3"));
		Assert.assertEquals("systemds.conf", m.get("$4"));
	}

	@Test(expected = ParseException.class)
	public void testBadArgs1() throws Exception {
		String cl = "systemds -f test.dml -args -config systemds.conf";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test
	public void testNVArgs1() throws Exception {
		String cl = "systemds -f test.dml -nvargs A=12 B=x.csv my123=12.2";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Map<String, String> m = o.argVals;
		Assert.assertEquals(3, m.size());
		Assert.assertEquals("12", m.get("$A"));
		Assert.assertEquals("x.csv", m.get("$B"));
		Assert.assertEquals("12.2", m.get("$my123"));
	}

	@Test(expected = ParseException.class)
	public void testBadNVArgs1() throws Exception {
		String cl = "systemds -f test.dml -nvargs";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test(expected = ParseException.class)
	public void testBadNVArgs2() throws Exception {
		String cl = "systemds -f test.dml -nvargs asd qwe";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test(expected = ParseException.class)
	public void testBadNVArgs3() throws Exception {
		String cl = "systemds -f test.dml -nvargs $X=12";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	@Test(expected = ParseException.class)
	public void testBadNVArgs4() throws Exception {
		String cl = "systemds -f test.dml -nvargs 123=123";
		String[] args = cl.split(" ");
		DMLOptions.parseCLArguments(args);
	}

	/**
	 * For Apache Commons CLI, if an argument to an option is enclosed in quotes,
	 * the leading and trailing quotes are stripped away. For instance, if the options is -arg and the
	 * argument is "foo"
	 *	-args "foo"
	 * Commons CLI will strip the quotes from "foo". This becomes troublesome when you really do
	 * want to pass in "foo" and not just foo.
	 * A way around this is to use 'foo` as done in {@link CLIOptionsParserTest#testNVArgs3()}
	 */
	@Test
	public void testNVArgs2() throws Exception {
		String cl = "systemds -f test.dml -args \"def\"";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Map<String, String> m = o.argVals;
		Assert.assertEquals("def", m.get("$1"));
	}


	/**
	 * See comment in {@link CLIOptionsParserTest#testNVArgs2()}
	 */
	@Test
	public void testNVArgs3() throws Exception {
		String cl = "systemds -f test.dml -args 'def'";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Map<String, String> m = o.argVals;
		Assert.assertEquals("'def'", m.get("$1"));
	}

	/**
	 * See comment in {@link CLIOptionsParserTest#testNVArgs2()}
	 * Additionally, if we try to pass something like
	 * -nvargs X="foo"
	 * Commons CLI will strip the leading and trailing quotes (viz. double quotes), which
	 * causes it to return
	 * X="foo
	 * The way to overcome this is to enclose the <value> of the <key=value> pair in single quotes
	 * and strip them away in the parsing code ourselves.
	 * TODO: Read the javadoc for this method, we can add in this logic if required
	 */
	@Test
	public void testNVArgs4() throws Exception {
		String cl = "systemds -f test.dml -nvargs abc='def'";
		String[] args = cl.split(" ");
		DMLOptions o = DMLOptions.parseCLArguments(args);
		Map<String, String> m = o.argVals;
		Assert.assertEquals("'def'", m.get("$abc"));
	}
}