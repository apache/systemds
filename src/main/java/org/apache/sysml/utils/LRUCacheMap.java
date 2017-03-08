package org.apache.sysml.utils;


import org.apache.sysml.runtime.DMLRuntimeException;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * An access ordered LRU Cache Map which conforms to the {@link Map} interface
 * while also providing the ability to get the least recently used entry
 * @param <K> the key type
 * @param <V> the value type
 */
public class LRUCacheMap<K,V> extends LinkedHashMap<K,V> {

  /**
   * Creates an access-ordered {@link LRUCacheMap}
   */
  public LRUCacheMap() {
    // An access-ordered LinkedHashMap is instantiated with the default initial capacity and load factors
    super(16, 0.75f, true);
  }

  // Private variables to assist in capturing the lease recently used entry
  private boolean evictLRU = false;
  private Map.Entry<K,V> lastEvictedEntry = null;

  /**
   * Removes and gets the least recently used entry
   * @return  the lease recently used entry
   * @throws DMLRuntimeException if the internal state is somehow corrupted
   */
  public Map.Entry<K,V> removeAndGetLRUEntry() throws DMLRuntimeException {
    lastEvictedEntry = null;
    if (size() <= 0){
      return null;
    }

    // The idea is to set removing the eldest entry to true and then putting in a dummy
    // entry (null, null). the removeEldestEntry will capture the eldest entry and is available
    // to return via a class member variable.
    evictLRU = true;
    V previous = super.put(null, null);
    remove(null);
    if (previous != null){
      throw new DMLRuntimeException("ERROR : Internal state of LRUCacheMap invalid - a value for the key 'null' is already present");
    }
    evictLRU = false;
    Map.Entry<K,V> toRet = lastEvictedEntry;
    return toRet;
  }

  @Override
  protected boolean removeEldestEntry(Map.Entry<K,V> eldest) {
    if (evictLRU) {
      lastEvictedEntry = eldest;
      return true;
    }
    return false;
  }

  @Override
  public V put (K k, V v){
    if (k == null)
      throw new IllegalArgumentException("ERROR: an entry with a null key was tried to be inserted in to the LRUCacheMap");
    return super.put (k, v);
  }


}
