package org.tugraz.sysds.common;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.conf.DMLConfig;

public class Warnings 
{
	private static final Log LOG = LogFactory.getLog(DMLConfig.class.getName());
	
	public static void warnFullFP64Conversion(long len) {
		LOG.warn("Performance warning: conversion to FP64 array of size "+len+".");
	}
	
	public static void warnInvalidBooleanIncrement(double delta) {
		LOG.warn("Correctness warning: invalid boolean increment by "+delta+".");
	}
}
