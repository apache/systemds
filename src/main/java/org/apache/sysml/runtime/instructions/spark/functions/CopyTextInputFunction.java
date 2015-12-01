/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.instructions.spark.functions;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

public class CopyTextInputFunction implements PairFunction<Tuple2<LongWritable, Text>,LongWritable, Text> 
{
	private static final long serialVersionUID = -196553327495233360L;

	public CopyTextInputFunction(  ) {
	
	}

	@Override
	public Tuple2<LongWritable, Text> call(
		Tuple2<LongWritable, Text> arg0) throws Exception {
		LongWritable lw = new LongWritable(arg0._1().get());
		Text txt = new Text(arg0._2());
		return new Tuple2<LongWritable,Text>(lw, txt);
	}
}