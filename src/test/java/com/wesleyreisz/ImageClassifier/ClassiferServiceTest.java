package com.wesleyreisz.ImageClassifier;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
//import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class ClassiferServiceTest {
    @Test
    void testImageClassification() {
        //load image
        BufferedImage bImage = null;
        try {
            ClassLoader classLoader = getClass().getClassLoader();
            File file = new File(classLoader.getResource("dog.jpg").getFile());
            bImage = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }

        assertEquals("cat", ClassiferService.classifyImage((bImage)));
    }

    /*@Test
    void testRunner() throws IOException {
        //Buffered stream reader to provide prompt requesting file path of
        // image to be tested
        //load image
        BufferedImage bImage = null;
        try {
            ClassLoader classLoader = getClass().getClassLoader();
            File file = new File(classLoader.getResource("lion.jpg").getFile());
            bImage = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Code to convert submitted image to INDArray
        // apply mean subtraction and run inference
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(bImage);
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);

        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        INDArray[] output = vgg16.output(false, image);
        System.out.println(output[0]);
    }*/
}