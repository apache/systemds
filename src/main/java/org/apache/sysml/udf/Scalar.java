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

package org.apache.sysml.udf;

/**
 * Class to represent a scalar input/output.
 * 
 * 
 * 
 */
public class Scalar extends FunctionParameter 
{
	

	private static final long serialVersionUID = 55239661026793046L;

	public enum ScalarValueType {
		Integer, Double, Boolean, Text
	};

	protected String _value;
	protected ScalarValueType _sType;

	/**
	 * Constructor to setup a scalar object.
	 * 
	 * @param t
	 * @param val
	 */
	public Scalar(ScalarValueType t, String val) {
		super(FunctionParameterType.Scalar);
		_sType = t;
		_value = val;
	}

	/**
	 * Method to get type of scalar.
	 * 
	 * @return
	 */
	public ScalarValueType getScalarType() {
		return _sType;
	}

	/**
	 * Method to get value for scalar.
	 * 
	 * @return
	 */
	public String getValue() {
		return _value;
	}

}
