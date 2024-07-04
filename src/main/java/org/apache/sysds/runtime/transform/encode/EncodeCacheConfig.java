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

import org.apache.sysds.conf.ConfigurationManager;

public class EncodeCacheConfig {

    private static boolean _cacheEnabled;
    private double _cacheMemoryFraction; // 5% of JVM heap size


    public static EncodeCacheConfig create() {
        // get the values from the dml script, and if they cannot be found, use the default values from this class
        try {
            final boolean enableCache = ConfigurationManager.isEncodeCacheEnabled();
            double memoryFraction = ConfigurationManager.getEncodeCacheMemoryFraction();
            return new EncodeCacheConfig(enableCache, memoryFraction);

        } catch (Exception e) { // use default values
            final boolean enableCache = true;
            double memoryFraction = 0.05;
            return new EncodeCacheConfig(enableCache, memoryFraction);
        }
    }

    private EncodeCacheConfig(boolean cacheEnabled, double cacheMemoryFraction) {
        this._cacheEnabled = cacheEnabled;
        this._cacheMemoryFraction = cacheMemoryFraction;
    }

    public enum EncodeCachePolicy {
        LRU
    }

    public double getCacheMemoryFraction() {
        return _cacheMemoryFraction;
    }

    public static boolean isCacheEnabled() {
        return _cacheEnabled;
    }

    public static EncodeCachePolicy _cachepolicy = EncodeCachePolicy.LRU;

    public void useCache(boolean use){
        _cacheEnabled = use;
    }

}
