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

package org.apache.sysds.runtime.compress.io;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ADictBasedColGroup;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.util.HDFSTool;

import scala.Tuple2;

public interface ReaderSparkCompressed {
    public static final Log LOG = LogFactory.getLog(ReaderSparkCompressed.class.getName());

    @SuppressWarnings("unchecked")
    public static JavaPairRDD<MatrixIndexes, MatrixBlock> getRDD(JavaSparkContext sc, String fileName) {

        final String dictName = fileName + ".dict";

        JavaPairRDD<MatrixIndexes, MatrixBlock> cmbRdd = sc
            .hadoopFile(fileName, SequenceFileInputFormat.class, MatrixIndexes.class, CompressedWriteBlock.class)
            .mapValues(new CompressUnwrap());

        if(HDFSTool.existsFileOnHDFS(dictName)) {

            JavaPairRDD<DictWritable.K, DictWritable> dictsRdd = sc.hadoopFile(dictName, SequenceFileInputFormat.class,
                DictWritable.K.class, DictWritable.class);

            return combineRdds(cmbRdd, dictsRdd);
        }
        else {
            return cmbRdd;
        }

    }

    private static JavaPairRDD<MatrixIndexes, MatrixBlock> combineRdds(JavaPairRDD<MatrixIndexes, MatrixBlock> cmbRdd,
        JavaPairRDD<DictWritable.K, DictWritable> dictsRdd) {
        // combine the elements
        JavaPairRDD<Integer, List<IDictionary>> dictsUnpacked = dictsRdd
            .mapToPair((t) -> new Tuple2<>(Integer.valueOf(t._1.id + 1), t._2.dicts));
        JavaPairRDD<Integer, Tuple2<MatrixIndexes, MatrixBlock>> mbrddC = cmbRdd
            .mapToPair((t) -> new Tuple2<>(Integer.valueOf((int) t._1.getColumnIndex()),
                new Tuple2<>(new MatrixIndexes(t._1), t._2)));

        return mbrddC.join(dictsUnpacked).mapToPair(ReaderSparkCompressed::combineTuples);

    }

    private static Tuple2<MatrixIndexes, MatrixBlock> combineTuples(
        Tuple2<Integer, Tuple2<Tuple2<MatrixIndexes, MatrixBlock>, List<IDictionary>>> e) {
        MatrixIndexes kOut = e._2._1._1;
        MatrixBlock mbIn = e._2._1._2;
        List<IDictionary> dictsIn = e._2._2;
        MatrixBlock ob = combineMatrixBlockAndDict(mbIn, dictsIn);
        return new Tuple2<>(new MatrixIndexes(kOut), ob);
    }

    private static MatrixBlock combineMatrixBlockAndDict(MatrixBlock mb, List<IDictionary> dicts) {
        if(mb instanceof CompressedMatrixBlock) {
            CompressedMatrixBlock cmb = (CompressedMatrixBlock) mb;
            List<AColGroup> gs = cmb.getColGroups();

            if(dicts.size() == gs.size()) {
                for(int i = 0; i < dicts.size(); i++) {

                    AColGroup g = gs.get(i);
                    if(g instanceof ADictBasedColGroup) {
                        gs.set(i, ((ADictBasedColGroup) g).copyAndSet(dicts.get(i)));
                    }
                }
            }
            else {
                int gis = 0;
                for(int i = 0; i < gs.size(); i++) {
                    AColGroup g = gs.get(i);
                    if(g instanceof ADictBasedColGroup) {
                        ADictBasedColGroup dg = (ADictBasedColGroup) g;
                        gs.set(i, dg.copyAndSet(dicts.get(gis)));
                        gis++;
                    }
                }
            }

            return new CompressedMatrixBlock(cmb.getNumRows(), cmb.getNumColumns(), cmb.getNonZeros(),
                cmb.isOverlapping(), gs);
        }
        else
            return mb;
    }
}
