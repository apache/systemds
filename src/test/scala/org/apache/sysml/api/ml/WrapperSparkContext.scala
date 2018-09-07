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

import org.scalatest.{ BeforeAndAfterAll, Suite }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.sql.SparkSession

trait WrapperSparkContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _
  @transient var sparkSession: SparkSession = _

  override def beforeAll() {
    super.beforeAll()
    sparkSession = SparkSession.builder().master("local[2]").appName("MLlibUnitTest").getOrCreate();
    sc = sparkSession.sparkContext;
  }

  override def afterAll() {
    try {
      sparkSession = null
      if (sc != null) {
        sc.stop()
      }
      sc = null
    } finally {
      super.afterAll()
    }
  }

}