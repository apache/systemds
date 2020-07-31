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

package org.apache.sysds.runtime.controlprogram.federated;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.sysds.api.DMLScript;

public class FederatedRequest implements Serializable {
	private static final long serialVersionUID = 5946781306963870394L;
	
	public enum FedMethod {
		READ_MATRIX, READ_FRAME, MATVECMULT, TRANSFER, AGGREGATE, SCALAR
	}
	
	private FedMethod _method;
	private List<Object> _data;
	private boolean _checkPrivacy;
	
	public FederatedRequest(FedMethod method, List<Object> data) {
		_method = method;
		_data = data;
		setCheckPrivacy();
	}
	
	public FederatedRequest(FedMethod method, Object ... datas) {
		_method = method;
		_data = Arrays.asList(datas);
		setCheckPrivacy();
	}
	
	public FederatedRequest(FedMethod method) {
		_method = method;
		_data = new ArrayList<>();
		setCheckPrivacy();
	}
	
	public FedMethod getMethod() {
		return _method;
	}
	
	public Object getParam(int i) {
		return _data.get(i);
	}
	
	public FederatedRequest appendParam(Object obj) {
		_data.add(obj);
		return this;
	}
	
	public FederatedRequest appendParams(Object ... objs) {
		_data.addAll(Arrays.asList(objs));
		return this;
	}
	
	public int getNumParams() {
		return _data.size();
	}
	
	public FederatedRequest deepClone() {
		return new FederatedRequest(_method, new ArrayList<>(_data));
	}

	public void setCheckPrivacy(boolean checkPrivacy){
		this._checkPrivacy = checkPrivacy;
	}

	public void setCheckPrivacy(){
		setCheckPrivacy(DMLScript.CHECK_PRIVACY);
	}

	public boolean checkPrivacy(){
		return _checkPrivacy;
	}
}
