package org.apache.sysml.api.ml.scala

import java.io.File

object ScriptsUtils {
  var systemmlHome = System.getenv("SYSTEMML_HOME")
  def resolvePath(filename:String):String = {
    import java.io.File
    ScriptsUtils.systemmlHome + File.separator + "algorithms" + File.separator + filename
  }
  def setSystemmlHome(path:String) {
    systemmlHome = path
  }
}