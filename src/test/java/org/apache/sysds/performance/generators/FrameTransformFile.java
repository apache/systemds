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

package org.apache.sysds.performance.generators;

import org.apache.sysds.performance.PerfUtil;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class FrameTransformFile extends ConstMatrix {

	final private String path;
	final private String specPath;

	private FrameTransformFile(String path, String specPath, MatrixBlock mb) throws Exception {
		super(mb);
		this.path = path;
		this.specPath = specPath;
	}

	// example:
	// src/test/resources/datasets/titanic/tfspec.json
	// src/test/resources/datasets/titanic/titanic.csv
	public static FrameTransformFile create(String path, String specPath) throws Exception {
		// read spec
		final String spec = PerfUtil.readSpec(specPath);
		final FrameFile fg = FrameFile.create(path);

		FrameBlock fb = fg.take();
		int k = InfrastructureAnalyzer.getLocalParallelism();
		FrameBlock sc = fb.detectSchema(k);
		fb = fb.applySchema(sc, k);
		MultiColumnEncoder encoder = EncoderFactory.createEncoder(spec, fb.getColumnNames(), fb.getNumColumns(), null);
		MatrixBlock mb = encoder.encode(fb, k);

		return new FrameTransformFile(path, specPath, mb);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" From file: ");
		sb.append(path);
		sb.append(" -- Transformed with: ");
		sb.append(specPath);
		return sb.toString();
	}

}
