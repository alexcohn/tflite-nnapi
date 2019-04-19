package org.tensorflow.lite.examples.classification.nnapicheck

import android.os.AsyncTask
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
       GlobalScope.launch() {
            async() {
                val classifier = Classifier(this@MainActivity)
                classifier.runAll()
            }
        }
    }
}
