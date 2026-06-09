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
import org.apache.sysds.runtime.transform.decode.*;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

/**
 * Unit test for validating dummy-coded column decoding in SystemDS.
 *
 * This test specifically verifies that columns encoded using dummy coding
 * (one-hot encoding) can be correctly decoded back to their original values
 * using {@link ColumnDecoderDummycode} through both:
 *  1. The legacy {@link Decoder} API
 *  2. The newer {@link ColumnDecoder} API (via {@link ColumnDecoderFactory}).
 */
public class ColumnDecoderDummycodeTest extends AutomatedTestBase {

    /**
     * Clears previous test assertions before running this test case.
     */
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    /**
     * Test decoding of a dummy-coded column.
     *
     * Steps performed:
     * 1. Generate a single-column frame of numeric categorical data.
     * 2. Apply dummy coding transformation (one-hot encoding).
     * 3. Decode using both:
     *    - Legacy {@link Decoder} API.
     *    - New {@link ColumnDecoder} API.
     * 4. Compare decoded frames to ensure correctness.
     */
    @Test
    public void testColumnDecoderDummycode() {
        try {
            // Step 1: Generate synthetic categorical data
            int rows = 20;

            // Fill column with repeating category values: 1, 2, 3, 1, 2, 3, ...
            double[][] arr = new double[rows][1];
            for (int i = 0; i < rows; i++)
                arr[i][0] = (i % 3) + 1;

            // Convert to MatrixBlock and FrameBlock
            MatrixBlock mb = DataConverter.convertToMatrixBlock(arr);
            FrameBlock data = DataConverter.convertToFrameBlock(mb);

            // Step 2: Define dummy coding transformation spec
            String spec = "{ids:true, dummycode:[1]}";

            // Step 3: Encode data using MultiColumnEncoder
            MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, null);
            MatrixBlock encoded = enc.encode(data);

            // Extract transformation metadata (category mapping)
            FrameBlock meta = enc.getMetaData(new FrameBlock(1, ValueType.STRING));

            // Step 4a: Decode using legacy Decoder API
            Decoder dec = DecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta, encoded.getNumColumns());
            FrameBlock expected = new FrameBlock(data.getSchema());
            dec.decode(encoded, expected);

            // Step 4b: Decode using new ColumnDecoder API
            ColumnDecoder cdec = ColumnDecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta);
            FrameBlock actual = new FrameBlock(data.getSchema());
            cdec.columnDecode(encoded, actual);

            // Step 5: Compare both decoded outputs
            TestUtils.compareFrames(expected, actual, false);
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}