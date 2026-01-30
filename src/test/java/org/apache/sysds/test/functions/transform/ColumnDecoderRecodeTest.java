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

/**
 * Unit test for validating recode column decoding in SystemDS.
 *
 * This test ensures that columns transformed with recoding
 * (mapping categorical/string values to numeric IDs)
 * can be correctly decoded back to their original values.
 *
 * It compares decoding outputs from the legacy {@link Decoder} API
 * and the newer {@link ColumnDecoder} API to verify equivalence.
 */
public class ColumnDecoderRecodeTest extends AutomatedTestBase {

    /**
     * Clears any previous assertion information before running tests.
     */
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    /**
     * Test recode decoding process using both Decoder and ColumnDecoder APIs.
     *
     * Steps:
     * 1. Create synthetic categorical column data (repeating values 1-4).
     * 2. Apply recode transformation using {@link MultiColumnEncoder}.
     * 3. Decode encoded data using:
     *    - Legacy {@link Decoder} API (baseline expected result).
     *    - New {@link ColumnDecoder} API.
     * 4. Compare decoded outputs for equality.
     */
    @Test
    public void testColumnEncoderDecoderRecode() {
        try {
            // Step 1: Generate categorical data
            int rows = 20;
            double[][] arr = new double[rows][1];

            // Fill single column with repeating sequence: 1, 2, 3, 4, 1, 2, ...
            for (int i = 0; i < rows; i++)
                arr[i][0] = (i % 4) + 1;

            // Convert array to SystemDS data structures
            MatrixBlock mb = DataConverter.convertToMatrixBlock(arr);
            FrameBlock data = DataConverter.convertToFrameBlock(mb);

            // Step 2: Define recode transformation spec
            String spec = "{ids:true, recode:[1]}";

            // Step 3: Encode data using MultiColumnEncoder
            MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, data.getColumnNames(), 1, null);
            MatrixBlock encoded = enc.encode(data);
            FrameBlock meta = enc.getMetaData(new FrameBlock(1, ValueType.STRING));

            // Step 4a: Decode using legacy Decoder API
            Decoder dec = DecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta, encoded.getNumColumns());
            FrameBlock expected = new FrameBlock(data.getSchema());
            dec.decode(encoded, expected);

            // Step 4b: Decode using new ColumnDecoder API
            ColumnDecoder cdec = ColumnDecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta);
            FrameBlock actual = new FrameBlock(data.getSchema());
            cdec.columnDecode(encoded, actual);

            // Step 5: Validate decoded outputs are identical
            TestUtils.compareFrames(expected, actual, false);
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}