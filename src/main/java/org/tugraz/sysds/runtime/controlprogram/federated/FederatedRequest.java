/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.controlprogram.federated;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FederatedRequest implements Serializable {
	private static final long serialVersionUID = 5946781306963870394L;
	
	public enum FedMethod {
		READ, MATVECMULT, TRANSFER, AGGREGATE, SCALAR
	}
	
	private FedMethod _method;
	private List<Object> _data;
	
	public FederatedRequest(FedMethod method, List<Object> data) {
		_method = method;
		_data = data;
	}
	
	public FederatedRequest(FedMethod method, Object ... datas) {
		_method = method;
		_data = Arrays.asList(datas);
	}
	
	public FederatedRequest(FedMethod method) {
		_method = method;
		_data = new ArrayList<>();
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
}
