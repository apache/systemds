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


package org.apache.sysml.runtime.matrix.data;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.sysml.runtime.util.FastStringTokenizer;


public class TextToBinaryCellConverter 
implements Converter<LongWritable, Text, MatrixIndexes, MatrixCell>
{
	
	private MatrixIndexes indexes = new MatrixIndexes();
	private MatrixCell value = new MatrixCell();
	private Pair<MatrixIndexes, MatrixCell> pair = new Pair<MatrixIndexes, MatrixCell>(indexes, value);
	private FastStringTokenizer st = new FastStringTokenizer(' '); 
	private boolean hasValue = false;
	private boolean toIgnore = false;

	@Override
	public void convert(LongWritable k1, Text v1) 
	{	
		String str = v1.toString();
		
		//handle support for matrix market format
		if(str.startsWith("%")) {
			if(str.startsWith("%%"))
				toIgnore=true;
			hasValue=false;
			return;
		}
		else if(toIgnore) {
			toIgnore=false;
			hasValue=false;
			return;
		}
			
		//reset the tokenizer
		st.reset( str );
		
		//convert text to matrix cell
		indexes.setIndexes( st.nextLong(), st.nextLong() );
		value.setValue( st.nextDouble() );
		hasValue = true;		
	}
	
	@Override
	public boolean hasNext() {
		return hasValue;
	}

	@Override
	public Pair<MatrixIndexes, MatrixCell> next() {
		if(!hasValue)
			return null;
		
		hasValue=false;
		return pair;
	}

	@Override
	public void setBlockSize(int rl, int cl) 
	{
		
	}
}
