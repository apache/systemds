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

package org.apache.sysds.runtime.transform.encode;

public class EncodeCacheEntry<T> { // uses generic to store any kind of object

    protected final EncodeCacheKey _key;
    protected T _value; // generic, can be an RCDMap or a BinBoundries object
    protected long _timestamp = 0;

    public EncodeCacheEntry(EncodeCacheKey key, T value) {
        _key = key;
        _value = value;
        _timestamp = System.currentTimeMillis();
    }

    protected synchronized void updateTimestamp(){
        _timestamp = System.currentTimeMillis();
    }

    public synchronized Object getValue() {
        return _value;
    }

    public long getSize() {
        if (_value instanceof BinMinsMaxs){ return ((BinMinsMaxs) _value).getSize(); }
        if (_value instanceof RCDMap) { return ((RCDMap) _value).getSize(); }
        else throw new RuntimeException("Cache entry does not contain bin boundaries or a recode map.");
    }

    public boolean isEmpty() { //
        return(_value == null);
    }

    @Override
    public String toString() {
        return "EncodeCacheEntry{" +
                "_key=" + _key +
                ", _value=" + _value +
                ", _timestamp=" + _timestamp +
                '}';
    }
}