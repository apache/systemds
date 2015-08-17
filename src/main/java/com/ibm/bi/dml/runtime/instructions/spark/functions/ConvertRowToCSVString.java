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
package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

public class ConvertRowToCSVString implements Function<Row, String> {
	private static final long serialVersionUID = -2399532576909402664L;

	@Override
	public String call(Row arg0) throws Exception {
		try {
			StringBuffer buf = new StringBuffer();
			if(arg0 != null) {
				for(int i = 0; i < arg0.length(); i++) {
					if(i > 0) {
						buf.append(",");
					}
					Double val = new Double(-1);
					try {
						val = Double.parseDouble(arg0.get(i).toString());
					}
					catch(Exception e) {
						throw new Exception("Only double types are supported as input to SystemML. The input argument is \'" + arg0.get(i) + "\'");
					}
					buf.append(val.toString());
				}
			}
			else {
				throw new Exception("Error while converting row to CSV string");
			}
			return buf.toString();
		}
		catch(Exception e) {
			throw new Exception("Error while converting row to CSV string:" + e.toString());
		}
	}

}
