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

package org.apache.sysds.runtime.privacy;

import java.util.HashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.BiFunction;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

public class CheckedConstraintsLog {
	private static HashMap<PrivacyLevel,LongAdder> checkedConstraintsTotal = new HashMap<PrivacyLevel,LongAdder>();
	private static BiFunction<LongAdder, LongAdder, LongAdder> mergeLongAdders = (v1, v2) -> {
		v1.add(v2.longValue() );
		return v1;
	};

	/**
	 * Adds checkedConstraints to the checked constraints total. 
	 * @param checkedConstraints constraints checked by federated worker
	 */
	public static void addCheckedConstraints(HashMap<PrivacyLevel,LongAdder> checkedConstraints){
		if ( checkedConstraints != null){
			checkedConstraints.forEach( 
			(key,value) -> checkedConstraintsTotal.merge( key, value, mergeLongAdders) );
		}
	}

	/**
	 * Remove all elements from checked constraints log.
	 */
	public static void reset(){
		checkedConstraintsTotal.clear();
	}

	public static HashMap<PrivacyLevel,LongAdder> getCheckedConstraints(){
		return checkedConstraintsTotal;
	}

	/**
	 * Get string representing all contents of the checked constraints log.
	 * @return string representation of checked constraints log.
	 */
	public static String display(){
		StringBuilder sb = new StringBuilder();
		checkedConstraintsTotal.forEach((k,v)->sb.append("\t" + k + ": " + v + "\n"));
		return sb.toString();
	}
}