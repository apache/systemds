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

package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class UncompressedScheme extends ACLAScheme {

    public UncompressedScheme(IColIndex cols) {
        super(cols);
    }

    @Override
    protected AColGroup encodeV(MatrixBlock data, IColIndex columns) {
        return ColGroupUncompressed.create(columns, data, false);
    }

    @Override
    protected AColGroup encodeVT(MatrixBlock data, IColIndex columns) {
        return ColGroupUncompressed.create(columns, data, true);
    }

    @Override
    protected ICLAScheme updateV(MatrixBlock data, IColIndex columns) {
        return this;
    }

    @Override
    protected ICLAScheme updateVT(MatrixBlock data, IColIndex columns) {
        return this;
    }

    @Override
    public UncompressedScheme clone() {
        return new UncompressedScheme(cols);
    }

}
