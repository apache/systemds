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

package org.apache.sysml.api.ml

import java.io.File
import java.io.BufferedReader
import java.io.InputStreamReader
import org.apache.sysml.runtime.DMLRuntimeException

object ScriptsUtils {
  var systemmlHome = System.getenv("SYSTEMML_HOME")

  /**
   * set SystemML home
   */
  def setSystemmlHome(path: String) {
    systemmlHome = path
  }
  
  /*
   * Internal function to get dml path
   */
  private[sysml] def resolvePath(scriptPath: String): String = {
    import java.io.File
    ScriptsUtils.systemmlHome + File.separator + scriptPath
  }

    /*
   * Internal function to get dml string from jar
   */
  private[sysml] def getDMLScript(scriptPath: String): String = {
    var reader: BufferedReader = null
    val out = new StringBuilder()
    try {
      val in = {
        if (systemmlHome == null || systemmlHome.equals("")) {
          classOf[LogisticRegression].getClassLoader().getResourceAsStream(scriptPath)
        } else {
          new java.io.FileInputStream(resolvePath(scriptPath))
        }
      }
      var reader = new BufferedReader(new InputStreamReader(in))
      var line = reader.readLine()
      while (line != null) {
        out.append(line);
        out.append(System.getProperty("line.separator"));
        line = reader.readLine()
      }
    } catch {
      case ex: Exception =>
        throw new DMLRuntimeException("Cannot read the script file " + scriptPath, ex)
    } finally {
      if (reader != null)
        reader.close();
    }
    out.toString()
  }
}