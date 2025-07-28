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
 * Unit test for validating pass-through column decoding in SystemDS.
 *
 * This test verifies that columns which are not transformed (i.e., no encoding applied)
 * are correctly decoded using {@link ColumnDecoderPassThrough}.
 *
 * It compares results from the legacy {@link Decoder} API and the new
 * {@link ColumnDecoder} API to ensure identical outputs.
 */
public class ColumnDecoderPassThroughTest extends AutomatedTestBase {

    /**
     * Clears previous assertion state before running each test.
     */
    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
    }

    /**
     * Test decoding for pass-through (untransformed) columns.
     *
     * Steps:
     * 1. Generate sequential numeric data.
     * 2. Use an empty transformation spec `{}` meaning no encoding/decoding is applied.
     * 3. Encode and decode using:
     *    - Legacy {@link Decoder} API.
     *    - New {@link ColumnDecoder} API.
     * 4. Compare decoded outputs to confirm correctness.
     */
    @Test
    public void testColumnDecoderPassThrough() {
        try {
            // Step 1: Generate sequential numeric data
            int rows = 20;
            MatrixBlock mb = MatrixBlock.seqOperations(1, rows, 1);
            FrameBlock data = DataConverter.convertToFrameBlock(mb);

            // Step 2: Define empty transformation spec (pass-through)
            String spec = "{}";

            // Step 3: Encode using MultiColumnEncoder
            MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, data.getColumnNames(), data.getNumColumns(), null);
            MatrixBlock encoded = enc.encode(data);
            FrameBlock meta = enc.getMetaData(new FrameBlock(data.getNumColumns(), ValueType.STRING));

            // Step 4a: Decode using legacy Decoder API
            Decoder dec = DecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta, encoded.getNumColumns());
            FrameBlock expected = new FrameBlock(data.getSchema());
            dec.decode(encoded, expected);

            // Step 4b: Decode using new ColumnDecoder API
            ColumnDecoder cdec = ColumnDecoderFactory.createDecoder(spec, data.getColumnNames(), data.getSchema(), meta);
            FrameBlock actual = new FrameBlock(data.getSchema());
            cdec.columnDecode(encoded, actual);

            // Step 5: Compare decoded outputs (expected vs actual)
            TestUtils.compareFrames(expected, actual, false);
        }
        catch(Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}