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

package org.apache.sysds.test.functions.transform;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.decode.ColumnDecoder;
import org.apache.sysds.runtime.transform.decode.ColumnDecoderFactory;
import org.apache.sysds.runtime.transform.decode.Decoder;
import org.apache.sysds.runtime.transform.decode.DecoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ColumnEncoderBinDecoderTest extends AutomatedTestBase {
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    @Test
    public void testColumnEncoderDecoderBin() {
        try {
            int rows = 20;
            MatrixBlock mb = MatrixBlock.seqOperations(1, rows, 1);
            FrameBlock data = DataConverter.convertToFrameBlock(mb);
            String spec = "{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}";

            // encode using column encoder
            MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, null);
            MatrixBlock encoded = enc.encode(data);
            FrameBlock meta = enc.getMetaData(new FrameBlock(1, ValueType.STRING));

            // baseline decode using existing decoder
            Decoder dec = DecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta, encoded.getNumColumns());
            FrameBlock expected = new FrameBlock(data.getSchema());
            dec.decode(encoded, expected);

            // decode using column decoder implementation
            ColumnDecoder cdec = ColumnDecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta);
            FrameBlock actual = new FrameBlock(data.getSchema());
            cdec.columnDecode(encoded, actual);

            TestUtils.compareFrames(expected, actual, false);
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}