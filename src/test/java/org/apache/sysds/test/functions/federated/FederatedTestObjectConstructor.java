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

package org.apache.sysds.test.functions.federated;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaData;
import org.junit.Assert;

public class FederatedTestObjectConstructor {
    public static MatrixObject constructFederatedInput(int rows, int cols, int blocksize, String host, long[][] begin,
        long[][] end, int[] ports, String[] inputs, String file) {
        MatrixObject fed = new MatrixObject(ValueType.FP64, file);
        try {
            fed.setMetaData(new MetaData(new MatrixCharacteristics(rows, cols, blocksize, rows * cols)));
            List<Pair<FederatedRange, FederatedData>> d = new ArrayList<>();
            for(int i = 0; i < ports.length; i++) {
                FederatedRange X1r = new FederatedRange(begin[i], end[i]);
                FederatedData X1d = new FederatedData(Types.DataType.MATRIX,
                    new InetSocketAddress(InetAddress.getByName(host), ports[i]), inputs[i]);
                d.add(new ImmutablePair<>(X1r, X1d));
            }

            InitFEDInstruction.federateMatrix(fed, d);
        }
        catch(Exception e) {
            e.printStackTrace();
            Assert.assertTrue(false);
        }
        return fed;

    }
}
