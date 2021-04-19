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

package org.apache.sysds.test.functions.io.libsvm;

import com.google.gson.Gson;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ReadFrameLIBSVMTest extends AutomatedTestBase {

  protected final static String TEST_DIR = "functions/io/libsvm/";
  private final static String TEST_NAME = "ReadFrameLIBSVMTest";
  protected final static String TEST_CLASS_DIR = TEST_DIR + ReadFrameLIBSVMTest.class.getSimpleName() + "/";

  protected String getInputLIBSVMFileName() {
    return "frame_" + getId();
  }

  protected int getId() {
    return 1;
  }

  protected String getTestClassDir() {
    return TEST_CLASS_DIR;
  }

  protected String getTestName() {
    return TEST_NAME;
  }

  @Override public void setUp() {
    TestUtils.clearAssertionInformation();
    addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Rout"}));
  }

  @Test public void testFrameLibsvm1_SP() {
    runlibsvmTest(1, ExecMode.SINGLE_NODE, false);
  }

  private void runlibsvmTest(int testNumber, ExecMode platform, boolean parallel) {
    ExecMode oldPlatform = rtplatform;
    rtplatform = platform;

    boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
    if(rtplatform == ExecMode.SPARK)
      DMLScript.USE_LOCAL_SPARK_CONFIG = true;

    boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;
    String output;
    try {
      CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;

      TestConfiguration config = getTestConfiguration(getTestName());

      loadTestConfiguration(config);

      String HOME = SCRIPT_DIR + TEST_DIR;
      String inputMatrixNameNoExtension = HOME + INPUT_DIR + getInputLIBSVMFileName();
      String inputMatrixNameWithExtension = inputMatrixNameNoExtension + ".libsvm";
      String dmlOutput = output("dml.scalar");

      fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
      programArgs = new String[] {"-args", inputMatrixNameWithExtension, dmlOutput};

      Gson gson = new Gson();
      System.out.println(gson.toJson(programArgs));

      output = runTest(true, false, null, -1).toString();

    }
    finally {
      rtplatform = oldPlatform;
      CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
      DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
    }
  }
}
