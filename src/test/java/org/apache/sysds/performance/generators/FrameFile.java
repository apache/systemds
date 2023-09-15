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

package org.apache.sysds.performance.generators;

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.meta.MetaDataAll;

public class FrameFile extends ConstFrame {

    final private String path;

    private FrameFile(String path, FrameBlock fb) {
        super(fb);
        this.path = path;
        System.out.println("First 10 rows:");
        System.out.println(fb.slice(0, 10));
    }

    public static FrameFile create(String path) throws Exception {

        MetaDataAll mba = new MetaDataAll(path + ".mtd", false, true);
        if(mba.mtdExists()) {
            LOG.error(mba);

            // DataCharacteristics ds = mba.getDataCharacteristics();
            FileFormat f = FileFormat.valueOf(mba.getFormatTypeString().toUpperCase());
            ValueType[] schema = FrameObject.parseSchema(mba.getSchema());
            FileFormatProperties p = null;
            if(f.equals(FileFormat.CSV)){
                p = new FileFormatPropertiesCSV();
                ((FileFormatPropertiesCSV)p).setHeader(mba.getHasHeader());
            }
            FrameReader r = FrameReaderFactory.createFrameReader(f, p);
            FrameBlock fb = r.readFrameFromHDFS(path, schema, mba.getDim1(), mba.getDim2());
            return new FrameFile(path, fb);
        }
        else {
            LOG.error("No Mtd file found.. please add one. Fallback to CSV reading with header");
            // we assume csv
            FrameReader r = FrameReaderFactory.createFrameReader(FileFormat.CSV);
            FrameBlock fb = r.readFrameFromHDFS(path, -1, -1);
            return new FrameFile(path, fb);
        }

    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append(" From file: ");
        sb.append(path);
        return sb.toString();
    }

}
