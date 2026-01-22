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

package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.sysds.runtime.compress.colgroup.ColGroupDDCLZW;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;

public abstract class DDCLZWScheme extends DDCScheme {
	// TODO: private int nUnique; Zu Datenspezifisch, überhaupt sinnvoll

	protected DDCLZWScheme(IColIndex cols) {
		super(cols);
	}

	public static DDCLZWScheme create(ColGroupDDCLZW g) {
		return g.getNumCols() == 1 ? new DDCLZWSchemeSC(g) : new DDCLZWSchemeMC(g);
	}

	public static DDCLZWScheme create(IColIndex cols) {
		return cols.size() == 1 ? new DDCLZWSchemeSC(cols) : new DDCLZWSchemeMC(cols);
	}

}
