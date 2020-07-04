package com.wesleyreisz.ImageClassifier;

import org.junit.jupiter.api.Test;

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

        assertEquals("Rottweiler", ClassiferService.classifyImage((bImage)));
    }

}