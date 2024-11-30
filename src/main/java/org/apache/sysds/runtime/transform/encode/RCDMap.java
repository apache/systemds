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

import java.util.Map;

public class RCDMap {

    private final Map<Object, Long> _rcdMap;
    private final long size;

    public RCDMap(Map<Object, Long> _rcdMap, long size) {
        this._rcdMap = _rcdMap;
        this.size = size;
    }

    public Map<Object, Long> get_rcdMap() {
        return _rcdMap;
    }

    public long getSize() {
        return size;
    }

    @Override
    public String toString() {
        return "RCDMap{" +
                "_rcdMap=" + _rcdMap +
                ", size=" + size +
                '}';
    }
}
