package org.apache.sysml.test.utils;

import org.apache.commons.cli.Options;
import org.apache.sysml.api.DMLScript;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;


public class CLIOptionsParserTest {

  private static Options options;

  @BeforeClass
  public static void setupOptions(){
    options = DMLScript.createCLIOptions();
  }

  @Test
  public void parseCLArguments1() throws Exception {
    String cl = "systemml -f test.dml";
    String[] args = cl.split(" ");
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals("test.dml", o.filePath);
  }

  @Test
  public void parseCLArguments2() throws Exception {
    String cl = "systemml -s \"print('hello world')\"";
    String[] args = cl.split(" ");
    DMLScript.DMLOptions o = DMLScript.parseCLArguments(args, options);
    Assert.assertEquals("print('hello world')", o.script);
  }

}