#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

import unittest

import pandas as pd
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, IndexToString
from pyspark.ml.regression import LinearRegression
import pyspark.sql.functions as sf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

import slicing.base.slicer as slicer
from slicing.base import union_slicer
from slicing.base.node import Node
from slicing.base.top_k import Topk
from slicing.spark_modules import spark_utils, spark_slicer, spark_union_slicer


class SliceTests(unittest.TestCase):
    loss_type = 0
    # x, y = m.generate_dataset(10, 100)
    train_dataset = pd.read_csv("toy_train.csv")
    attributes_amount = len(train_dataset.values[0])
    model = linear_model.LinearRegression()
    y_train = train_dataset.iloc[:, attributes_amount - 1:attributes_amount].values
    x_train = train_dataset.iloc[:, 0:attributes_amount - 1].values
    model.fit(x_train, y_train)
    test_dataset = pd.read_csv("toy.csv")
    y_test = test_dataset.iloc[:, attributes_amount - 1:attributes_amount].values
    x_test = test_dataset.iloc[:, 0:attributes_amount - 1].values
    y_pred = model.predict(x_test)
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    print('r_2 statistic: %.2f' % r2_score(y_test, y_pred))
    # Now that we have trained the model, we can print the coefficient of x that it has predicted
    print('Coefficients: \n', model.coef_)
    enc = OneHotEncoder(handle_unknown='ignore')
    x = enc.fit_transform(x_test).toarray()
    complete_x = []
    complete_y = []
    counter = 0
    for item in x:
        complete_x.append((counter, item))
        complete_y.append((counter, y_test[counter]))
        counter = counter + 1
    all_features = enc.get_feature_names()
    loss = mean_squared_error(y_test, y_pred)
    devs = (y_pred - y_test) ** 2
    errors = []
    counter = 0
    for pred in devs:
        errors.append((counter, pred))
        counter = counter + 1
    k = 5
    w = 0.5
    alpha = 4
    top_k = Topk(k)
    debug = True
    b_update = True
    first_level = slicer.make_first_level(all_features, list(complete_x), loss, len(complete_x), y_test, errors,
                                          loss_type, top_k, alpha, w)
    first_level_nodes = first_level[0]
    slice_member = first_level_nodes[(7, 'x2_2')]

    def test_attr_spark(self):
        conf = SparkConf().setAppName("toy_test").setMaster('local[2]')
        num_partitions = 2
        enumerator = "join"
        model_type = "regression"
        label = 'target'
        sparkContext = SparkContext(conf=conf)
        sqlContext = SQLContext(sparkContext)
        train_df = sqlContext.read.csv("toy_train.csv", header='true',
                            inferSchema='true')
        test_df = sqlContext.read.csv("toy.csv", header='true',
                            inferSchema='true')
        # initializing stages of main transformation pipeline
        stages = []
        # list of categorical features for further hot-encoding
        cat_features = ['a', 'b', 'c']
        for feature in cat_features:
            string_indexer = StringIndexer(inputCol=feature, outputCol=feature + "_index").setHandleInvalid("skip")
            encoder = OneHotEncoderEstimator(inputCols=[string_indexer.getOutputCol()], outputCols=[feature + "_vec"])
            encoder.setDropLast(False)
            stages += [string_indexer, encoder]
        assembler_inputs = [feature + "_vec" for feature in cat_features]
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="assembled_inputs")
        stages += [assembler]
        assembler_final = VectorAssembler(inputCols=["assembled_inputs"], outputCol="features")
        stages += [assembler_final]
        pipeline = Pipeline(stages=stages)
        train_pipeline_model = pipeline.fit(train_df)
        test_pipeline_model = pipeline.fit(test_df)
        train_df_transformed = train_pipeline_model.transform(train_df)
        test_df_transformed = test_pipeline_model.transform(test_df)
        train_df_transformed = train_df_transformed.withColumn('model_type', sf.lit(0))
        test_df_transformed = test_df_transformed.withColumn('model_type', sf.lit(0))
        decode_dict = {}
        counter = 0
        cat = 0
        for feature in cat_features:
            colIdx = test_df_transformed.select(feature, feature + "_index").distinct().rdd.collectAsMap()
            colIdx = {k: v for k, v in sorted(colIdx.items(), key=lambda item: item[1])}
            for item in colIdx:
                decode_dict[counter] = (cat, item, colIdx[item], counter)
                counter = counter + 1
            cat = cat + 1
        train_df_transform_fin = train_df_transformed.select('features', label, 'model_type')
        test_df_transform_fin = test_df_transformed.select('features', label, 'model_type')
        lr = LinearRegression(featuresCol='features', labelCol=label, maxIter=10, regParam=0.0, elasticNetParam=0.8)
        lr_model = lr.fit(train_df_transform_fin)
        eval = lr_model.evaluate(test_df_transform_fin)
        f_l2 = eval.meanSquaredError
        pred = eval.predictions
        pred_df_fin = pred.withColumn('error', spark_utils.calc_loss(pred[label], pred['prediction'], pred['model_type']))
        predictions = pred_df_fin.select('features', 'error').repartition(num_partitions)
        converter = IndexToString(inputCol='features', outputCol='cats')
        all_features = list(decode_dict)
        predictions = predictions.collect()
        spark_join = spark_slicer.parallel_process(all_features, predictions, f_l2, sparkContext, debug=self.debug, alpha=self.alpha,
                                      k=self.k, w=self.w, loss_type=self.loss_type, enumerator="join")
        spark_union = spark_union_slicer.process(all_features, predictions, f_l2, sparkContext, debug=self.debug, alpha=self.alpha,
                                      k=self.k, w=self.w, loss_type=self.loss_type, enumerator="union")
        self.assertEqual(3, len(spark_join.slices))
        print("check1")
        self.assertEqual(spark_join.min_score, spark_union.min_score)
        print("check2")
        self.assertEqual(spark_join.keys, spark_union.keys)
        print("check3")
        self.assertEqual(len(spark_join.slices), len(spark_union.slices))
        print("check4")
        idx = -1
        for sliced in spark_join.slices:
            idx += 1
            self.assertEqual(sliced.score, spark_union.slices[idx].score)
        print("check5")

    def test_features_number(self):
        self.assertEqual(len(self.all_features), 9)
        print("check 1")

    def test_base_first_level(self):
        self.assertEqual(9, len(self.first_level_nodes))
        print("check 2")

    def test_parents_first(self):
        self.assertIn(('x2_2', 7), self.slice_member.parents)
        print("check 3")

    def test_name(self):
        self.assertEqual('x2_2', self.slice_member.make_name())
        print("check 4")

    def test_size(self):
        self.assertEqual(36, self.slice_member.size)
        print("check 5")

    def test_e_upper(self):
        self.assertEqual(81, self.slice_member.e_upper)
        print("check 6")

    def test_loss(self):
        self.assertEqual(22, int(self.slice_member.loss))
        print("check 7")

    def test_opt_fun(self):
        self.slice_member.score = slicer.opt_fun(self.slice_member.loss, self.slice_member.size, self.loss, len(self.x_test), self.w)
        print("check 8")

    def test_score(self):
        self.assertEqual(1.2673015873015872, self.slice_member.score)
        print("check 9")

    def test_base_join_enum(self):
        cur_lvl_nodes = {}
        all_nodes = {}
        b_update = True
        cur_lvl = 1
        slice_index = (2, 'x0_3')
        combined = slicer.join_enum(slice_index, self.first_level_nodes, self.complete_x, self.loss,
                               len(self.complete_x), self.y_test, self.errors, self.debug, self.alpha, self.w,
                               self.loss_type, b_update, cur_lvl, all_nodes, self.top_k, cur_lvl_nodes)
        self.assertEqual(6, len(combined[0]))
        print("check1")

    def test_parents_second(self):
        cur_lvl_nodes = {}
        all_nodes = {}
        b_update = True
        cur_lvl = 1
        slice_index = (2, 'x0_3')
        combined = slicer.join_enum(slice_index, self.first_level_nodes, self.complete_x, self.loss,
                                    len(self.complete_x), self.y_test, self.errors, self.debug, self.alpha, self.w,
                                    self.loss_type, b_update, cur_lvl, all_nodes, self.top_k, cur_lvl_nodes)
        parent1 = combined[0][('x0_3 && x1_3')]
        parent2 = combined[0][('x0_3 && x2_2')]
        new_node = Node(self.complete_x, self.loss, len(self.complete_x), self.y_test, self.errors)
        new_node.parents = [parent1, parent2]
        parent1_attr = parent1.attributes
        parent2_attr = parent2.attributes
        new_node_attr = slicer.union(parent1_attr, parent2_attr)
        self.assertEqual(new_node_attr, [('x0_3', 2), ('x1_3', 5), ('x2_2', 7)])
        print("check2")

    def test_nonsense(self):
        cur_lvl_nodes = {}
        all_nodes = {}
        b_update = True
        cur_lvl = 1
        slice_index = (2, 'x0_3')
        combined = slicer.join_enum(slice_index, self.first_level_nodes, self.complete_x, self.loss,
                                    len(self.complete_x), self.y_test, self.errors, self.debug, self.alpha, self.w,
                                    self.loss_type, b_update, cur_lvl, all_nodes, self.top_k, cur_lvl_nodes)
        parent1 = combined[0][('x0_3 && x1_3')]
        parent2 = combined[0][('x0_3 && x2_2')]
        new_node = Node(self.complete_x, self.loss, len(self.complete_x), self.y_test, self.errors)
        new_node.parents = [parent1, parent2]
        parent1_attr = parent1.attributes
        parent2_attr = parent2.attributes
        new_node_attr = slicer.union(parent1_attr, parent2_attr)
        new_node.attributes = new_node_attr
        new_node.name = new_node.make_name()
        flagTrue = slicer.slice_name_nonsense(parent1, parent2, 2)
        self.assertEqual(True, flagTrue)
        print("check3")

    def test_non_nonsense(self):
        cur_lvl_nodes = {}
        all_nodes = {}
        b_update = True
        cur_lvl = 1
        slice_index = (2, 'x0_3')
        parent3 = Node(self.complete_x, self.loss, len(self.complete_x), self.y_test, self.errors)
        parent3.parents = [self.first_level_nodes[(4, 'x1_2')], self.first_level_nodes[(7, 'x2_2')]]
        parent3.attributes = [('x1_2', 4), ('x2_2', 7)]
        combined = slicer.join_enum(slice_index, self.first_level_nodes, self.complete_x, self.loss,
                                    len(self.complete_x), self.y_test, self.errors, self.debug, self.alpha, self.w,
                                    self.loss_type, b_update, cur_lvl, all_nodes, self.top_k, cur_lvl_nodes)
        parent2 = combined[0]['x0_3 && x2_3']
        parent3.key = (8, 'x1_2 && x2_2')
        flag_nonsense = slicer.slice_name_nonsense(parent2, parent3, 2)
        self.assertEqual(True, flag_nonsense)
        print("check4")

    def test_uppers(self):
        cur_lvl_nodes = {}
        all_nodes = {}
        b_update = True
        cur_lvl = 1
        slice_index = (2, 'x0_3')
        parent3 = Node(self.complete_x, self.loss, len(self.complete_x), self.y_test, self.errors)
        parent3.parents = [self.first_level_nodes[(4, 'x1_2')], self.first_level_nodes[(7, 'x2_2')]]
        parent3.attributes = [('x1_2', 4), ('x2_2', 7)]
        combined = slicer.join_enum(slice_index, self.first_level_nodes, self.complete_x, self.loss,
                                    len(self.complete_x), self.y_test, self.errors, self.debug, self.alpha, self.w,
                                    self.loss_type, b_update, cur_lvl, all_nodes, self.top_k, cur_lvl_nodes)
        parent1 = combined[0]['x0_3 && x1_3']
        parent2 = combined[0]['x0_3 && x2_3']
        new_node = Node(self.complete_x, self.loss, len(self.complete_x), self.y_test, self.errors)
        new_node.parents = [parent1, parent2]
        new_node.calc_bounds(2, self.w)
        self.assertEqual(25, new_node.s_upper)
        print("check5")
        self.assertEqual(398, int(new_node.c_upper))
        print("check6")

    def test_topk_slicing(self):
        join_top_k = slicer.process(self.all_features, self.complete_x, self.loss, len(self.complete_x), self.y_test, self.errors,
                       self.debug, self.alpha, self.k, self.w, self.loss_type, self.b_update)
        union_top_k = union_slicer.process(self.all_features, self.complete_x, self.loss, len(self.complete_x), self.y_test, self.errors,
                       self.debug, self.alpha, self.k, self.w, self.loss_type, self.b_update)
        self.assertEqual(join_top_k.min_score, union_top_k.min_score)
        print("check1")
        self.assertEqual(join_top_k.keys, union_top_k.keys)
        print("check2")
        self.assertEqual(len(join_top_k.slices), len(union_top_k.slices))
        print("check3")
        idx = -1
        for sliced in join_top_k.slices:
            idx += 1
            self.assertEqual(sliced.score, union_top_k.slices[idx].score)
        print("check4")

    def test_extreme_target(self):
        test_dataset = pd.read_csv("/home/lana/diploma/project/slicing/datasets/toy_extreme_change.csv")
        y_test = test_dataset.iloc[:, self.attributes_amount - 1:self.attributes_amount].values
        x_test = test_dataset.iloc[:, 0:self.attributes_amount - 1].values
        y_pred = self.model.predict(x_test)
        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, y_pred))
        print('r_2 statistic: %.2f' % r2_score(y_test, y_pred))
        # Now that we have trained the model, we can print the coefficient of x that it has predicted
        print('Coefficients: \n', self.model.coef_)
        enc = OneHotEncoder(handle_unknown='ignore')
        x = enc.fit_transform(x_test).toarray()
        complete_x = []
        complete_y = []
        counter = 0
        for item in x:
            complete_x.append((counter, item))
            complete_y.append((counter, y_test[counter]))
            counter = counter + 1
        all_features = enc.get_feature_names()
        loss = mean_squared_error(y_test, y_pred)
        devs = (y_pred - y_test) ** 2
        errors = []
        counter = 0
        for pred in devs:
            errors.append((counter, pred))
            counter = counter + 1
        k = 5
        w = 0.5
        alpha = 4
        top_k = Topk(k)
        debug = True
        b_update = True
        first_level = slicer.make_first_level(all_features, list(complete_x), loss, len(complete_x), y_test, errors,
                                              self.loss_type, top_k, alpha, w)
        first_level_nodes = first_level[0]
        slice_member = first_level_nodes[(7, 'x2_2')]
        self.assertGreater(slice_member.loss, self.slice_member.loss)
        print("check 1")
        self.assertGreater(slice_member.score, self.slice_member.score)
        print("check 2")

    def test_error_significance(self):
        y_test = self.test_dataset.iloc[:, self.attributes_amount - 1:self.attributes_amount].values
        x_test = self.test_dataset.iloc[:, 0:self.attributes_amount - 1].values
        y_pred = self.model.predict(x_test)
        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, y_pred))
        print('r_2 statistic: %.2f' % r2_score(y_test, y_pred))
        # Now that we have trained the model, we can print the coefficient of x that it has predicted
        print('Coefficients: \n', self.model.coef_)
        enc = OneHotEncoder(handle_unknown='ignore')
        x = enc.fit_transform(x_test).toarray()
        complete_x = []
        complete_y = []
        counter = 0
        for item in x:
            complete_x.append((counter, item))
            complete_y.append((counter, y_test[counter]))
            counter = counter + 1
        all_features = enc.get_feature_names()
        loss = mean_squared_error(y_test, y_pred)
        devs = (y_pred - y_test) ** 2
        errors = []
        counter = 0
        for pred in devs:
            errors.append((counter, pred))
            counter = counter + 1
        k = 5
        # Maximized size significance
        w = 0
        alpha = 4
        top_k = Topk(k)
        debug = True
        b_update = True
        first_level = slicer.make_first_level(all_features, list(complete_x), loss, len(complete_x), y_test, errors,
                                              self.loss_type, top_k, alpha, w)
        first_level_nodes = first_level[0]
        slice_member = first_level_nodes[(7, 'x2_2')]
        self.assertGreater(self.slice_member.score, slice_member.score)

    def test_size_significance(self):
        y_test = self.test_dataset.iloc[:, self.attributes_amount - 1:self.attributes_amount].values
        x_test = self.test_dataset.iloc[:, 0:self.attributes_amount - 1].values
        y_pred = self.model.predict(x_test)
        print("Mean squared error: %.2f"
                  % mean_squared_error(y_test, y_pred))
        print('r_2 statistic: %.2f' % r2_score(y_test, y_pred))
        # Now that we have trained the model, we can print the coefficient of x that it has predicted
        print('Coefficients: \n', self.model.coef_)
        enc = OneHotEncoder(handle_unknown='ignore')
        x = enc.fit_transform(x_test).toarray()
        complete_x = []
        complete_y = []
        counter = 0
        for item in x:
            complete_x.append((counter, item))
            complete_y.append((counter, y_test[counter]))
            counter = counter + 1
        all_features = enc.get_feature_names()
        loss = mean_squared_error(y_test, y_pred)
        devs = (y_pred - y_test) ** 2
        errors = []
        counter = 0
        for pred in devs:
            errors.append((counter, pred))
            counter = counter + 1
        k = 5
        # Maximized size significance
        w = 1
        alpha = 4
        top_k = Topk(k)
        debug = True
        b_update = True
        first_level = slicer.make_first_level(all_features, list(complete_x), loss, len(complete_x), y_test, errors,
                                                  self.loss_type, top_k, alpha, w)
        first_level_nodes = first_level[0]
        slice_member = first_level_nodes[(7, 'x2_2')]
        self.assertGreater(slice_member.score, self.slice_member.score)


if __name__ == '__main__':
    unittest.main()
