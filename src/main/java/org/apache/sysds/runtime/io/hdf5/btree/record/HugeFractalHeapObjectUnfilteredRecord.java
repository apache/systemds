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

package org.apache.sysds.runtime.io.hdf5.btree.record;

import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.nio.ByteBuffer;

public class HugeFractalHeapObjectUnfilteredRecord extends BTreeRecord {
    private final long hugeObjectAddress;
    private final long hugeObjectLength;
    private final long hugeObjectID;

    public HugeFractalHeapObjectUnfilteredRecord(ByteBuffer bb) {
        if (bb.remaining() < 24) {
            throw new HdfException("Invalid length buffer for "
        		+ "HugeFractalHeapObjectUnfilteredRecord. remaining bytes = " + bb.remaining());
        }

        hugeObjectAddress = Utils.readBytesAsUnsignedLong(bb, 8);
        hugeObjectLength = Utils.readBytesAsUnsignedLong(bb, 8);
        hugeObjectID = Utils.readBytesAsUnsignedLong(bb, 8);
    }

    public long getAddress() {
        return this.hugeObjectAddress;
    }

    public long getLength() {
        return this.hugeObjectLength;
    }

    public long getId() {
        return this.hugeObjectID;
    }
}

