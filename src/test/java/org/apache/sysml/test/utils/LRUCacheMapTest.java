package org.apache.sysml.test.utils;

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