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

import org.apache.sysml.utils.LRUCacheMap;
import org.junit.Assert;
import org.junit.Test;

import java.util.Map;

public class LRUCacheMapTest {

  @Test
  public void test1() throws Exception {
    LRUCacheMap<String, Long> m = new LRUCacheMap<String, Long>();
    m.put("k1", 10l);
    m.put("k2", 20l);
    m.put("k3", 30l);
    m.put("k4", 40l);

    Map.Entry<String, Long> e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k1", e.getKey());
  }

  @Test
  public void test2() throws Exception {
    LRUCacheMap<String, Long> m = new LRUCacheMap<String, Long>();
    m.put("k1", 10l);
    m.put("k2", 20l);
    m.put("k3", 30l);
    m.put("k4", 40l);
    m.get("k1");

    Map.Entry<String, Long> e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k2", e.getKey());
  }

  @Test(expected = IllegalArgumentException.class)
  public void test3() {
    LRUCacheMap<String, Long> m = new LRUCacheMap<String, Long>();
    m.put(null, 10l);
  }

  @Test
  public void test4() throws Exception {
    LRUCacheMap<String, Long> m = new LRUCacheMap<String, Long>();
    m.put("k1", 10l);
    m.put("k2", 20l);
    m.put("k3", 30l);
    m.put("k4", 40l);
    m.remove("k1");
    m.remove("k2");

    Map.Entry<String, Long> e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k3", e.getKey());
  }

  @Test
  public void test5() throws Exception {
    LRUCacheMap<String, Long> m = new LRUCacheMap<String, Long>();
    m.put("k1", 10l);
    m.put("k2", 20l);
    m.put("k1", 30l);

    Map.Entry<String, Long> e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k2", e.getKey());
  }

  @Test
  public void test6() throws Exception {
    LRUCacheMap<String, Long> m = new LRUCacheMap<String, Long>();
    m.put("k1", 10l);
    m.put("k2", 20l);
    m.put("k3", 30l);
    m.put("k4", 40l);
    m.put("k5", 50l);
    m.put("k6", 60l);
    m.put("k7", 70l);
    m.put("k8", 80l);
    m.get("k4");


    Map.Entry<String, Long> e;
    e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k1", e.getKey());
    e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k2", e.getKey());
    e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k3", e.getKey());
    e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k5", e.getKey());
    e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k6", e.getKey());
    e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k7", e.getKey());
    e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k8", e.getKey());
    e = m.removeAndGetLRUEntry();
    Assert.assertEquals("k4", e.getKey());

  }


}