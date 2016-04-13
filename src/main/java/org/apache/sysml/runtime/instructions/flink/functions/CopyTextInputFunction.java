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

package org.apache.sysml.runtime.instructions.flink.functions;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

public class CopyTextInputFunction implements MapFunction<Tuple2<LongWritable, Text>, Tuple2<LongWritable, Text>> {

	private static final long serialVersionUID = -196553327495233360L;

	public CopyTextInputFunction() {
	}

	@Override
	public Tuple2<LongWritable, Text> map(Tuple2<LongWritable, Text> arg0) throws Exception {
		LongWritable lw = new LongWritable(arg0.f0.get());
		Text txt = new Text(arg0.f1);
		return new Tuple2<LongWritable, Text>(lw, txt);
	}
}
