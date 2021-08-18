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

package org.apache.sysds.runtime.iogen;

import com.google.gson.Gson;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.*;

public class GenerateReader {

	public static MatrixReader generateReader(String sampleRaw, MatrixBlock sampleMatrix) throws Exception {
		MatrixReader reader = null;

		ReaderMapping rp = new ReaderMapping(sampleRaw, sampleMatrix);

		boolean isMapped = rp.isMapped();
		if(!isMapped) {
			throw new Exception("Sample raw data and sample matrix don't match !!");
		}

		//////////////////////////////////////////////////
		System.out.println("Mapped !!!!!!!!!!!!");
		Gson gson = new Gson();
		System.out.println("Map Row >> " + gson.toJson(rp.getMapRow()));
		System.out.println("Map Col >> " + gson.toJson(rp.getMapCol()));
		System.out.println("Map Size >> " + gson.toJson(rp.getMapSize()));

		FileFormatProperties ffp =rp.getFormatProperties();
		if(ffp!=null){
			System.out.println(gson.toJson(ffp));
		}
		else
			throw new Exception("The file format couldn't recognize!!");

		return reader;
	}
}
