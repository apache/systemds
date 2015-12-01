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

package org.apache.sysml.parser;

import org.apache.sysml.runtime.util.UtilFunctions;



public class DoubleIdentifier extends ConstIdentifier 
{
	
	private double _val;
	
	
	public DoubleIdentifier(double val, String filename, int blp, int bcp, int elp, int ecp){
		super();
		 _val = val;
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.DOUBLE);
        this.setAllPositions(filename, blp, bcp, elp, ecp);
	}
	
	public DoubleIdentifier(DoubleIdentifier d, String filename, int blp, int bcp, int elp, int ecp){
		super();
		 _val = d.getValue();
		_kind = Kind.Data;
		this.setDimensions(0,0);
        this.computeDataType();
        this.setValueType(ValueType.DOUBLE);
        this.setAllPositions(filename, blp, bcp, elp, ecp);
	}
	
	public Expression rewriteExpression(String prefix) throws LanguageException{
		return this;
	}
	
	// Used only by the parser for unary operation
	public void multiplyByMinusOne() {
		_val = -1 * _val;
	}
	
	public double getValue(){
		return _val;
	}
	
	public void setValue(double v) {
		_val = v;
	}
	
	public String toString(){
		return Double.toString(_val);
	}
	
	@Override
	public VariableSet variablesRead() {
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
		return null;
	}
	
	@Override
	public long getLongValue() {
		return UtilFunctions.toLong(getValue());
	}
}
