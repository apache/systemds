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

import org.apache.spark.api.java.function.PairFunction;

import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

import scala.Tuple2;

public class CopyBinaryCellFunction implements PairFunction<Tuple2<MatrixIndexes, MatrixCell>,MatrixIndexes, MatrixCell> 
{

	private static final long serialVersionUID = -675490732439827014L;

	public CopyBinaryCellFunction(  ) {
	
	}

	@Override
	public Tuple2<MatrixIndexes, MatrixCell> call(
		Tuple2<MatrixIndexes, MatrixCell> arg0) throws Exception {
		MatrixIndexes ix = new MatrixIndexes(arg0._1());
		MatrixCell cell = new MatrixCell();
		cell.copy(arg0._2());
		return new Tuple2<MatrixIndexes,MatrixCell>(ix,cell);
	}
}