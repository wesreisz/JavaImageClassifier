package com.wesleyreisz.ImageClassifier;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;
class StringUtilsTest {

    @Test
    void testImageToByteArray() {
        BufferedImage bImage = loadImage();
        byte[] tensor = StringUtils.Convert2ByteArray(bImage);
        assertEquals("ffffffff", Integer.toHexString(tensor[0]));
    }

    @Test
    void cropImageToRect() {
        BufferedImage bImage = loadImage();

        BufferedImage cropppedBImage = StringUtils.cropImageToRect(bImage);
        assertNotNull(cropppedBImage);
    }

    @Test
    void scaleImage() {
        BufferedImage bImage = loadImage();
        BufferedImage cropppedBImage = StringUtils.scaleImage(bImage,299, 299);
        assertNotNull(cropppedBImage);
        assertTrue((cropppedBImage.toString().contains("height = 299")));
    }


    @Test
    void makeImageTensor() {
        BufferedImage bImage = loadImage();
        BufferedImage croppedBImage = StringUtils.cropImageToRect(bImage);
        BufferedImage scaledImage = StringUtils.scaleImage(croppedBImage,299, 299);
        INDArray test = StringUtils.makeImageTensor(scaledImage,0, (float)299);
        assertNotNull(test);
    }

    private BufferedImage loadImage(){
        BufferedImage bImage = null;
        try {
            ClassLoader classLoader = getClass().getClassLoader();
            File file = new File(classLoader.getResource("lion.jpg").getFile());
            bImage = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bImage;
    }
}