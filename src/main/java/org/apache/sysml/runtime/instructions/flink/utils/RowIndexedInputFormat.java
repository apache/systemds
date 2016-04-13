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

package org.apache.sysml.runtime.instructions.flink.utils;

import org.apache.flink.api.common.io.DelimitedInputFormat;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.fs.FileInputSplit;

import java.io.IOException;
import java.nio.charset.Charset;

public class RowIndexedInputFormat extends DelimitedInputFormat<Tuple2<Integer, String>> {


	private static final long serialVersionUID = 1L;
	private String charsetName = "UTF-8";

	public void open(FileInputSplit split) throws IOException {
		super.open(split);
	}

	public Tuple2<Integer, String> readRecord(Tuple2<Integer, String> reuseable, byte[] bytes, int offset,
											  int numBytes) throws IOException {
		if (this.getDelimiter() != null && this.getDelimiter().length == 1 && this.getDelimiter()[0] == 10 && offset + numBytes >= 1 && bytes[offset + numBytes - 1] == 13) {
			--numBytes;
		}
		return new Tuple2<Integer, String>(this.currentSplit.getSplitNumber(),
				new String(bytes, offset, numBytes, this.charsetName));
	}

	public String getCharsetName() {
		return this.charsetName;
	}

	public void setCharsetName(String charsetName) {
		if (charsetName == null) {
			throw new IllegalArgumentException("Charset must not be null.");
		} else {
			this.charsetName = charsetName;
		}
	}

	public void configure(Configuration parameters) {
		super.configure(parameters);
		if (this.charsetName == null || !Charset.isSupported(this.charsetName)) {
			throw new RuntimeException("Unsupported charset: " + this.charsetName);
		}
	}

	public String toString() {
		return "TextInputFormat (" + this.getFilePath() + ") - " + this.charsetName;
	}
}
