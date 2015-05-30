from py4j.protocol import Py4JJavaError, Py4JError
import sys, traceback
from pyspark.sql import *

# Simple wrapper class for Python
class MLContext(object):
	def __init__(self, sc):
		if isinstance(sc, basestring):
			try:
				self.ml = MLContext(sc)
			except Py4JError:
				print "Execution type not supported"
		else:
			try:
				self.sc = sc
				self.ml = sc._jvm.com.ibm.bi.dml.api.MLContext(sc._jsc)
			except Py4JError:
				print "Error: PySpark cannot find SystemML.jar"

	def execute(self, *args):
		numArgs = len(args)
		print numArgs
		
		# One Argument means only DML script
		if numArgs == 1:
			try:
				jmlOut = self.ml.execute(args[0])
				mlOut = MLOutput(jmlOut)
				return mlOut
			except Py4JJavaError:
				traceback.print_exc()

		# Two arguments, script + either Array of Strings of Boolean
		elif numArgs == 2:
			try:
				jmlOut = self.ml.execute(args[0], args[1])
				mlOut = MLOutput(jmlOut)
				return mlOut
			except Py4JJavaError:
				traceback.print_exc()

		# Three arguemtns
		elif numArgs == 3:
			try:
                                jmlOut = self.ml.execute(args[0], args[1], args[2])
       	                        mlOut = MLOutput(jmlOut)
               	                return mlOut
			except Py4JJavaError:
                               	traceback.print_exc()
		else:
			raise TypeError("Arguments do not match MLContext-API")


	def registerInput(self, varName, df):
		try:
			self.ml.registerInput(varName, df._jdf)
		except Py4JJavaError:
			traceback.print_exc()

	def registerOutput(self, varName):
		try:
                        self.ml.registerOutput(varName)
                except Py4JJavaError:
                        traceback.print_exc()

class MLOutput(object):
	def __init__(self, jmlOut):
		self.jmlOut = jmlOut

	def getDF(self, sqlContext, varName):
		try:
			jdf = self.jmlOut.getDF(sqlContext._scala_SQLContext, varName)
			df = DataFrame(jdf, sqlContext)
			return df
		except Py4JJavaError:
                        traceback.print_exc()

	def getExplainOutput(self):
		try:
			return self.jmlOut.getExplainOutput()
		except Py4JJavaError:
                        traceback.print_exc()
