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

package org.apache.sysds.test.functions.builtin.part2;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class BuiltinSTEPGlmTest extends AutomatedTestBase 
{
  private final static String TEST_NAME = "stepGLM";
  private final static String TEST_DIR = "functions/builtin/";
  private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSTEPGlmTest.class.getSimpleName() + "/";

  @Override
  public void setUp() {
    addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{}));
  }

  @Test
  public void testLmMatrixDenseCPlm() {
    runSTEPGlmTest(ExecType.CP);
  }

  @Test
  public void testLmMatrixSparseSPlm() {
    runSTEPGlmTest(ExecType.SPARK);
  }

  private void runSTEPGlmTest(ExecType instType) {
    ExecMode platformOld = setExecMode(instType);

    try {
      loadTestConfiguration(getTestConfiguration(TEST_NAME));

      String HOME = SCRIPT_DIR + TEST_DIR;

      // Pointing to the generated validation DML script
      fullDMLScriptName = HOME + TEST_NAME + ".dml";
      programArgs = new String[]{};

      // runTest executes the script; fails if the DML script invokes stop()
      runTest(true, false, null, -1);
    }
    finally {
      rtplatform = platformOld;
    }
  }
}
