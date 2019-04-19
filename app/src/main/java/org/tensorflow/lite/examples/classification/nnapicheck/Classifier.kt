/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.classification.nnapicheck

import android.app.Activity
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import androidx.core.graphics.blue
import androidx.core.graphics.green
import androidx.core.graphics.red
import kotlinx.android.synthetic.main.activity_main.*

import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter

import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

/** A classifier specialized to label images using TensorFlow Lite.  */
class Classifier {

    companion object {
        private val TAG = "Classifier"
    }

    private fun fillFloatBuffer(buffer: ByteBuffer): Boolean {
        buffer.rewind()
        buffer.order(ByteOrder.nativeOrder())
        for (n in 0 until buffer.limit()/12) {
            buffer.putFloat(200f)
            buffer.putFloat(200f)
            buffer.putFloat(200f)
        }
        return true
    }

    private val decodeResults = HashMap<Int, Any>()

    private val shapeBuffer = ByteBuffer.allocateDirect(1 * 180 * 320 * 3 * 4)
    private val imgData = ByteBuffer.allocateDirect(1* 360 * 640 * 3 * 4)
    private val magnification_alpha = ByteBuffer.allocateDirect(4)

    private val magnifiedImage = Array(1) { Array(360) { Array(640) { FloatArray(3) } } }

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    /** Optional GPU delegate for accleration.  */
    private var gpuDelegate: Delegate? = null

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    protected var tflite: Interpreter? = null

    val modelPath_float = "dummy_decoder_model_360_640_3_float.tflite"

    private var activity: Activity

    /** Initializes a `Classifier`.  */
    @Throws(IOException::class)
    internal constructor(activity: Activity) {

        this.activity = activity
        decodeResults[0] = magnifiedImage

        fillFloatBuffer(magnification_alpha)
        fillFloatBuffer(imgData)
        fillFloatBuffer(shapeBuffer)
    }

    /** Memory-map the model file in Assets.  */
    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity, modelPath: String): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /** Runs inference and returns the classification results.  */
    private fun run(): String {
        val startTime = SystemClock.uptimeMillis()
        runInference()
        val endTime = SystemClock.uptimeMillis()
        Log.d(TAG, "Timecost to run model inference: ${endTime - startTime}")
        return (endTime - startTime).toString()
    }

    fun runAll() {

        Log.d(TAG, "Run Tensorflow Lite " + org.tensorflow.lite.TensorFlowLite.runtimeVersion() + ":" + org.tensorflow.lite.TensorFlowLite.schemaVersion());

        var tfliteModel = loadModelFile(activity, modelPath_float)
        tfliteOptions.setUseNNAPI(false)
        tflite = Interpreter(tfliteModel, tfliteOptions)
        var res = run()
        activity.runOnUiThread {
            activity.text_result.append("CPU: ${res} ms\n")
        }

        tfliteOptions.setUseNNAPI(true)
        tflite = Interpreter(tfliteModel, tfliteOptions)
        try {
            res = run()
        }
        catch (ex: RuntimeException) {
            res = ex.message!!
        }
        activity.runOnUiThread {
            activity.text_result.append("NNAPI: ${res} ms\n")
        }
    }

    /** Enables use of the GPU for inference, if available.  */
    fun useGpu() {
//        if (gpuDelegate == null && GpuDelegateHelper.isGpuDelegateAvailable()) {
//            gpuDelegate = GpuDelegateHelper.createGpuDelegate()
//            tfliteOptions.addDelegate(gpuDelegate)
//            recreateInterpreter()
//        }
    }

    /** Closes the interpreter and model to release resources.  */
    fun close() {
        if (tflite != null) {
            tflite!!.close()
            tflite = null
        }
    }

    protected fun runInference() {
        tflite?.runForMultipleInputsOutputs(arrayOf(shapeBuffer, imgData, magnification_alpha), decodeResults)
    }
}
