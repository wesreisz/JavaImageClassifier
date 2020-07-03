package com.wesleyreisz.ImageClassifier;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.awt.image.BufferedImage;
import java.io.IOException;

public class ClassiferService {
    public static String classifyImage(BufferedImage input) {
        //load the model
        ComputationGraph model = importModel("xception.h5");

        //crop, scale, and set RBG on image
        BufferedImage croppedImage = StringUtils.cropImageToRect(input);
        BufferedImage scaledImage = StringUtils.scaleImage(croppedImage, 299, 299);

        //this is the correct input shape to the model (I have to get the image into this shape)
        //INDArray dummyData = Nd4j.create(new int[] {1, 299, 299, 3});
        INDArray data = StringUtils.makeImageTensor(scaledImage, 128, 128);

        System.out.println("min:" + data.min());
        System.out.println("max:" + data.max());

        INDArray result = model.output(data)[0]; // selects the only output tensor.
        System.out.println("Test:" + result);
        System.out.println("Size of output: " + result.length());
        System.out.println("Class index:" + result.argMax(1)); //this is an answer
        System.out.println("Max activation:" + result.max(1));
        return "cat";
    }

    private static ComputationGraph importModel(String modelName) {
        // load the model
        String simpleMlp = null;
        try {
            simpleMlp = new ClassPathResource(modelName).getFile().getPath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        ComputationGraph graph = null;
        try {
            try {
                int[] inputShape = new int[]{299, 299, 3};
                KerasModelBuilder builder = new KerasModel()
                        .modelBuilder()
                        .inputShape(inputShape)
                        .modelHdf5Filename(simpleMlp)
                        .enforceTrainingConfig(false);
                KerasModel model = builder.buildModel();
                graph = model.getComputationGraph();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (InvalidKerasConfigurationException e) {
            e.printStackTrace();
        } catch (UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
        }
        return graph;
    }
}
