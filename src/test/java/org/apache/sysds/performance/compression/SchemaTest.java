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
package org.apache.sysds.performance.compression;

import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.scheme.CompressionScheme;
import org.apache.sysds.runtime.compress.lib.CLALibScheme;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class SchemaTest extends APerfTest<Object, MatrixBlock> {

	public SchemaTest(int N, IGenerate<MatrixBlock> gen) {
		super(N, gen);
	}

	public void run() throws Exception, InterruptedException {
		System.out.println(this);
		execute(() -> sumTask(), "Sum Task -- Warmup");
		execute(() -> compressTask(), "Compress Normal 10 blocks");
		final CompressedMatrixBlock cmb = (CompressedMatrixBlock) ret.get(0);
		final CompressionScheme sch = CLALibScheme.getScheme(cmb);
		execute(() -> updateScheme(sch), "Update Scheme");
		execute(() -> applyScheme(sch), "Apply Scheme");
		final CompressionScheme sch2 = CLALibScheme.getScheme(cmb);
		execute(() -> updateAndApplyScheme(sch2), "Update & Apply Scheme");
		execute(() -> fromEmptyScheme(), "From Empty Update & Apply Scheme");
	}

	public void runCom() throws Exception, InterruptedException {
		execute(() -> compressTaskDoNotKeep(), "Compress Normal 10 blocks", 10);
		for(int i = 0; i < 100; i++)
			execute(() -> fromEmptySchemeDoNotKeep(), "From Empty Update & Apply Scheme", 10000);
		
	}

	protected String makeResString() {
		return "";
	}

	private void sumTask() {
		gen.take().sum();
	}

	private void compressTask() {
		ret.add(CompressedMatrixBlockFactory.compress(gen.take()).getLeft());
	}

	private void compressTaskDoNotKeep() {
		CompressedMatrixBlockFactory.compress(gen.take()).getLeft();
	}

	private void updateScheme(CompressionScheme sch) {
		sch.update(gen.take());
	}

	private void applyScheme(CompressionScheme sch) {
		ret.add(sch.encode(gen.take()));
	}

	private void updateAndApplyScheme(CompressionScheme sch) {
		MatrixBlock mb = gen.take();
		sch.update(mb);
		ret.add(sch.encode(mb));
	}

	private void fromEmptyScheme() {
		MatrixBlock mb = gen.take();
		CompressionScheme sch = CLALibScheme.genScheme(CompressionType.EMPTY, mb.getNumColumns());
		sch.update(mb);
		ret.add(sch.encode(mb));
	}

	private void fromEmptySchemeDoNotKeep() {
		MatrixBlock mb = gen.take();
		CompressionScheme sch = CLALibScheme.genScheme(CompressionType.EMPTY, mb.getNumColumns());
		
		sch.update(mb);
		sch.encode(mb);
	}
}
