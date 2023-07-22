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
package org.apache.sysds.runtime.instructions.spark.functions;

import java.util.Iterator;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.sysds.runtime.matrix.data.BinaryBlockToTextCellConverter;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

import scala.Tuple2;

public class ConvertMatrixBlockToIJVLines implements FlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, String> {

	private static final long serialVersionUID = 3555147684480763957L;
	
	int blen;
	public ConvertMatrixBlockToIJVLines(int blen) {
		this.blen = blen;
	}
	
	@Override
	public Iterator<String> call(Tuple2<MatrixIndexes, MatrixBlock> kv) {
		final BinaryBlockToTextCellConverter converter = new BinaryBlockToTextCellConverter();
		converter.setBlockSize(blen, blen);
		converter.convert(kv._1, kv._2);
		
		Iterable<String> ret = new Iterable<>() {
			@Override
			public Iterator<String> iterator() {
				return new Iterator<>() {
					
					@Override
					public void remove() {}
					
					@Override
					public String next() {
						return converter.next().getValue().toString();
					}
					
					@Override
					public boolean hasNext() {
						return converter.hasNext();
					}
				};
			}
		};
		
		return ret.iterator();
	}

}
