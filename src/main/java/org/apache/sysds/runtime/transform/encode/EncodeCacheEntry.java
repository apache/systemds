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

import org.apache.sysds.runtime.transform.encode.EncodeCacheConfig.EncodeCacheStatus;

import java.util.HashMap;

public class EncodeCacheEntry<T> { // uses generic to store any kind of object

    protected final EncodeCacheKey _key; //TODO: do we need it in here too?
    protected T _value; // generic, can be an RCDMap or a BinBoundries object
    protected long _timestamp = 0;
    protected EncodeCacheConfig.EncodeCacheStatus _status; //TODO: do we need this?

    public EncodeCacheEntry(EncodeCacheKey key, T value) {
        _key = key;
        _value = value;
        _timestamp = System.currentTimeMillis();
        _status = isEmpty() ? EncodeCacheStatus.EMPTY : EncodeCacheStatus.CACHED;
    }

    //TODO: not sure about the signature here
    protected synchronized void setCacheStatus(EncodeCacheStatus st) {
        _status = st;
    }

    protected synchronized void updateTimestamp(){
        _timestamp = System.currentTimeMillis();
    }

    public synchronized Object getValue() {
        return _value;
    }

    //TODO: Copied the following from the lineageCacheEntry but not clear whether we need them

    /*public synchronized void setValue(T val) {
        _value = val;
        _status = isEmpty() ? EncodeCacheStatus.EMPTY : EncodeCacheStatus.CACHED;
        //resume all threads waiting for val
        notifyAll();
    }*/

    /*protected synchronized void removeAndNotify() {
        //Set the status to NOTCACHED (not cached anymore) and wake up the sleeping threads
        if (_status != EncodeCacheStatus.EMPTY)
            return;
        _status = EncodeCacheConfig.EncodeCacheStatus.NOTCACHED;
        notifyAll();
    }*/

    public long getSize() {
        if (_value instanceof BinMinsMaxs){ return ((BinMinsMaxs) _value).getSize(); }
        if (_value instanceof RCDMap) { return ((RCDMap) _value).getSize(); }
        else throw new RuntimeException("Cache entry does not contain bin boundaries or a recode map.");
    }

    public boolean isEmpty() { //
        return(_value == null);
    }










}
