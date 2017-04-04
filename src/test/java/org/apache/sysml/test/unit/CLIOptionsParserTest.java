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

package org.apache.sysml.test.unit;

import java.util.Map;

import org.apache.commons.cli.AlreadySelectedException;
import org.apache.commons.cli.MissingOptionException;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.mlcontext.ScriptType;
import org.apache.sysml.utils.Explain;
import org.junit.Assert;
import org.junit.Test;


public class CLIOptionsParserTest {

  @Test(expected = MissingOptionException.class)
  public void testNoOptions() throws Exception {
    String cl = "systemml";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test
  public void testFile() throws Exception {
    String cl = "systemml -f test.dml";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals("test.dml", o.filePath);
    Assert.assertEquals(ScriptType.DML, o.scriptType);

  }

  @Test
  public void testScript() throws Exception {
    String cl = "systemml -s \"print('hello')\"";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals("print('hello')", o.script);
  }

  @Test
  public void testConfig() throws Exception {
    String cl = "systemml -s \"print('hello')\" -config SystemML-config.xml";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals("print('hello')", o.script);
    Assert.assertEquals("SystemML-config.xml", o.configFile);
  }

  @Test
  public void testDebug() throws Exception {
    String cl = "systemml -s \"print('hello')\" -debug";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals("print('hello')", o.script);
    Assert.assertEquals(true, o.debug);
  }

  @Test
  public void testClean() throws Exception {
    String cl = "systemml -clean";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(true, o.clean);
  }

  @Test(expected = AlreadySelectedException.class)
  public void testBadClean() throws Exception {
    String cl = "systemml -clean -f test.dml";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test(expected = AlreadySelectedException.class)
  public void testBadScript() throws Exception {
    String cl = "systemml -f test.dml -s \"print('hello')\"";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test
  public void testStats() throws Exception {
    String cl = "systemml -f test.dml -stats";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(true, o.stats);
    Assert.assertEquals(10, o.statsCount);
  }

  @Test
  public void testStatsCount() throws Exception {
    String cl = "systemml -f test.dml -stats 9123";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(true, o.stats);
    Assert.assertEquals(9123, o.statsCount);
  }

  @Test(expected = ParseException.class)
  public void testBadStats() throws Exception {
    String cl = "systemml -f test.dml -stats help";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(true, o.stats);
  }

  @Test
  public void testGPUForce() throws Exception {
    String cl = "systemml -f test.dml -gpu force";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(true, o.gpu);
    Assert.assertEquals(true, o.forceGPU);
  }

  @Test(expected = ParseException.class)
  public void testBadGPUOption() throws Exception {
    String cl = "systemml -f test.dml -gpu f2orce";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test
  public void testPython() throws Exception {
    String cl = "systemml -f test.dml -python";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(ScriptType.PYDML, o.scriptType);
  }

  @Test
  public void testHelp() throws Exception {
    String cl = "systemml -help";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(true, o.help);
  }

  @Test(expected = AlreadySelectedException.class)
  public void testBadHelp() throws Exception {
    String cl = "systemml -help -clean";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(true, o.help);
  }

  @Test
  public void testExplain1() throws Exception {
    String cl = "systemml -f test.dml -explain";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(Explain.ExplainType.RUNTIME, o.explainType);
  }

  @Test
  public void testExplain2() throws Exception {
    String cl = "systemml -f test.dml -explain hops";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(Explain.ExplainType.HOPS, o.explainType);
  }

  @Test
  public void testExplain3() throws Exception {
    String cl = "systemml -f test.dml -explain runtime";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(Explain.ExplainType.RUNTIME, o.explainType);
  }

  @Test
  public void testExplain4() throws Exception {
    String cl = "systemml -f test.dml -explain recompile_hops";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(Explain.ExplainType.RECOMPILE_HOPS, o.explainType);
  }

  @Test
  public void testExplain5() throws Exception {
    String cl = "systemml -f test.dml -explain recompile_runtime";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(Explain.ExplainType.RECOMPILE_RUNTIME, o.explainType);
  }

  @Test
  public void testExec1() throws Exception {
    String cl = "systemml -f test.dml -exec hadoop";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(DMLScript.RUNTIME_PLATFORM.HADOOP, o.execMode);
  }

  @Test
  public void testExec2() throws Exception {
    String cl = "systemml -f test.dml -exec spark";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(DMLScript.RUNTIME_PLATFORM.SPARK, o.execMode);
  }

  @Test
  public void testExec3() throws Exception {
    String cl = "systemml -f test.dml -exec singlenode";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(DMLScript.RUNTIME_PLATFORM.SINGLE_NODE, o.execMode);
  }

  @Test
  public void testExec4() throws Exception {
    String cl = "systemml -f test.dml -exec hybrid";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(DMLScript.RUNTIME_PLATFORM.HYBRID, o.execMode);
  }

  @Test
  public void testExec5() throws Exception {
    String cl = "systemml -f test.dml -exec hybrid_spark";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals(DMLScript.RUNTIME_PLATFORM.HYBRID_SPARK, o.execMode);
  }

  @Test(expected = ParseException.class)
  public void testBadExec() throws Exception {
    String cl = "systemml -f test.dml -exec new_system";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test
  public void testArgs1() throws Exception {
    String cl = "systemml -f test.dml -args 10 \"x.csv\"";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Map<String, String> m = o.argVals;
    Assert.assertEquals(2, m.size());
    Assert.assertEquals("10", m.get("$1"));
    Assert.assertEquals("x.csv", m.get("$2"));
  }

  @Test
  public void testArgs2() throws Exception {
    String cl = "systemml -f test.dml -args 10 \"x.csv\" 1234.2 systemml.conf -config systemml.conf";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Map<String, String> m = o.argVals;
    Assert.assertEquals(4, m.size());
    Assert.assertEquals("10", m.get("$1"));
    Assert.assertEquals("x.csv", m.get("$2"));
    Assert.assertEquals("1234.2", m.get("$3"));
    Assert.assertEquals("systemml.conf", m.get("$4"));
  }

  @Test(expected = ParseException.class)
  public void testBadArgs1() throws Exception {
    String cl = "systemml -f test.dml -args -config systemml.conf";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test
  public void testNVArgs1() throws Exception {
    String cl = "systemml -f test.dml -nvargs A=12 B=x.csv my123=12.2";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Map<String, String> m = o.argVals;
    Assert.assertEquals(3, m.size());
    Assert.assertEquals("12", m.get("$A"));
    Assert.assertEquals("x.csv", m.get("$B"));
    Assert.assertEquals("12.2", m.get("$my123"));
  }

  @Test(expected = ParseException.class)
  public void testBadNVArgs1() throws Exception {
    String cl = "systemml -f test.dml -nvargs";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test(expected = ParseException.class)
  public void testBadNVArgs2() throws Exception {
    String cl = "systemml -f test.dml -nvargs asd qwe";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test(expected = ParseException.class)
  public void testBadNVArgs3() throws Exception {
    String cl = "systemml -f test.dml -nvargs $X=12";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  @Test(expected = ParseException.class)
  public void testBadNVArgs4() throws Exception {
    String cl = "systemml -f test.dml -nvargs 123=123";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.parseCLArguments(args, options);
  }

  /**
   * For Apache Commons CLI, if an argument to an option is enclosed in quotes,
   * the leading and trailing quotes are stripped away. For instance, if the options is -arg and the
   * argument is "foo"
   *  -args "foo"
   * Commons CLI will strip the quotes from "foo". This becomes troublesome when you really do
   * want to pass in "foo" and not just foo.
   * A way around this is to use 'foo` as done in {@link CLIOptionsParserTest#testNVArgs3()}
   */
  @Test
  public void testNVArgs2() throws Exception {
    String cl = "systemml -f test.dml -args \"def\"";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Map<String, String> m = o.argVals;
    Assert.assertEquals("def", m.get("$1"));
  }


  /**
   * See comment in {@link CLIOptionsParserTest#testNVArgs2()}
   */
  @Test
  public void testNVArgs3() throws Exception {
    String cl = "systemml -f test.dml -args 'def'";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
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
    String cl = "systemml -f test.dml -nvargs abc='def'";
    String[] args = cl.split(" ");
    Options options = DMLScript.createCLIOptions();
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Map<String, String> m = o.argVals;
    Assert.assertEquals("'def'", m.get("$abc"));
  }

}