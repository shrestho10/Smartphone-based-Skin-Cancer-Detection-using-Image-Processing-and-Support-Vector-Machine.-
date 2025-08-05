package com.example.opencv002;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;


import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
//import android.support.annotation.Nullable;

import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.photo.Photo;
import org.opencv.utils.Converters;

import android.view.View;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;


import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static org.opencv.core.Core.bitwise_and;
import static org.opencv.core.Core.bitwise_not;
import static org.opencv.core.Core.bitwise_or;
import static org.opencv.core.Core.bitwise_xor;
import static org.opencv.core.Core.convertScaleAbs;
import static org.opencv.core.Core.findNonZero;
import static org.opencv.core.Core.flip;
import static org.opencv.core.Core.invert;
import static org.opencv.core.Core.mean;
import static org.opencv.core.Core.transpose;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_NONE;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2BGRA;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.arcLength;
import static org.opencv.imgproc.Imgproc.boundingRect;
import static org.opencv.imgproc.Imgproc.connectedComponents;
import static org.opencv.imgproc.Imgproc.contourArea;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.fitEllipse;
import static org.opencv.imgproc.Imgproc.getStructuringElement;
import static org.opencv.imgproc.Imgproc.minAreaRect;
import static org.opencv.imgproc.Imgproc.minEnclosingCircle;
import static org.opencv.imgproc.Imgproc.moments;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.imgproc.Imgproc.threshold;



public class MainActivity extends AppCompatActivity {
    // UI components
    ImageView imageView;
    TextView textview1;

    // Image variables
    Uri imageuri;
    Bitmap newbitmap;

    // For displaying image name
    String sab1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textview1 = findViewById(R.id.textView2);

        // Initialize OpenCV
        OpenCVLoader.initDebug();
    }

    // Open Gallery to select an image
    public void OpenGallery(View v) {
        Intent myIntent = new Intent(Intent.ACTION_GET_CONTENT, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(myIntent, 100);
    }

    // Handle the result from image picker
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 100 && resultCode == RESULT_OK && data != null) {
            imageuri = data.getData();
            String sab = imageuri.toString();
            String[] parts = sab.split("F");
            sab1 = parts[parts.length - 1];

            try {
                newbitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageuri);
            } catch (IOException e) {
                System.out.println("Image Insertion Problem");
            }

            textview1.refreshDrawableState();
            imageView.setImageBitmap(newbitmap);
        }
    }

    // Perform Otsu thresholding and analysis
    public void otsu(View v) {
        try {
            // Convert Bitmap to Mat
            Mat imageMat = new Mat();       // For geometrical analysis
            Mat imageMat2 = new Mat();      // For color-based processing
            Utils.bitmapToMat(newbitmap, imageMat);
            Utils.bitmapToMat(newbitmap, imageMat2);

            // Resize images for uniform processing
            Size resizeSize = new Size(600, 600);
            Imgproc.resize(imageMat, imageMat, resizeSize);
            Imgproc.resize(imageMat2, imageMat2, resizeSize);

            // Structuring element for morphological operations
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));

            // Color image preprocessing (for hair/noise removal)
            Imgproc.GaussianBlur(imageMat2, imageMat2, new Size(5, 5), 0);
            Imgproc.morphologyEx(imageMat2, imageMat2, Imgproc.MORPH_CLOSE, kernel);

            // Geometric preprocessing
            Imgproc.GaussianBlur(imageMat, imageMat, new Size(5, 5), 0);
            Imgproc.morphologyEx(imageMat, imageMat, Imgproc.MORPH_CLOSE, kernel);
            Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_RGB2GRAY); // Convert to grayscale

            // Apply Otsu thresholding
            Imgproc.threshold(imageMat, imageMat, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
            Core.bitwise_not(imageMat, imageMat); // Invert binary image

            // Find contours
            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(imageMat, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

            // Find largest contour by area
            double maxArea = 0;
            MatOfPoint maxContour = new MatOfPoint();
            int firstContourCounter = 0, firstContourIndex = 0;

            for (MatOfPoint contour : contours) {
                firstContourCounter++;
                double area = Imgproc.contourArea(contour);
                if (area > maxArea) {
                    maxArea = area;
                    firstContourIndex = firstContourCounter;
                    maxContour = contour;
                }
            }

            // Calculate centroid using image moments
            Moments m = Imgproc.moments(maxContour);
            Point centroid = new Point();
            centroid.x = m.get_m10() / m.get_m00();
            centroid.y = m.get_m01() / m.get_m00();

            // Calculate perimeter and circularity
            double perimeter = Imgproc.arcLength(new MatOfPoint2f(maxContour.toArray()), true);
            double circularity = (4 * Math.PI * maxArea) / (perimeter * perimeter);

            // Calculate average distance of boundary points from centroid
            double edgeDistanceSum = 0;
            int edgePointCount = 0;

            for (int i = 0; i < maxContour.rows(); i++) {
                for (int j = 0; j < maxContour.cols(); j++) {
                    double x = maxContour.get(i, j)[0];
                    double y = maxContour.get(i, j)[1];

                    double dx = centroid.x - x;
                    double dy = centroid.y - y;
                    double distance = Math.sqrt(dx * dx + dy * dy);

                    edgeDistanceSum += distance;
                    edgePointCount++;
                }
            }


// Calculate mean distance from center to contour points
double edgeDistanceMean = valueofedge / edgecounter;

double individualEdgeDist = 0;
double totalEdgeDistVariance = 0;

for (int row = 0; row < max_contour.rows(); row++) {
    for (int col = 0; col < max_contour.cols(); col++) {
        double x = max_contour.get(row, col)[0];
        double y = max_contour.get(row, col)[1];

        double dx = centroid.x - x;
        double dy = centroid.y - y;
        double distance = Math.sqrt(dx * dx + dy * dy);

        individualEdgeDist = Math.pow(distance - edgeDistanceMean, 2);
        totalEdgeDistVariance += individualEdgeDist;
    }
}

// Edge abruptness calculation (variation relative to perimeter and mean distance)
double edgeAbruptness = (totalEdgeDistVariance / epsilon) / Math.pow(edgeDistanceMean, 2);

// Ellipse fitting
RotatedRect fittedEllipse;
double majorAxis, minorAxis;

if (max_contour.toArray().length >= 5) {
    fittedEllipse = Imgproc.fitEllipse(new MatOfPoint2f(max_contour.toArray()));
    majorAxis = fittedEllipse.size.height;
    minorAxis = fittedEllipse.size.width;

    if (majorAxis < minorAxis) {
        double temp = majorAxis;
        majorAxis = minorAxis;
        minorAxis = temp;
    }
} else {
    fittedEllipse = Imgproc.minAreaRect(new MatOfPoint2f(max_contour.toArray()));
    majorAxis = fittedEllipse.size.height;
    minorAxis = fittedEllipse.size.width;

    if (minorAxis > majorAxis) {
        double temp = majorAxis;
        majorAxis = minorAxis;
        minorAxis = temp;
    }
}

// Rotate image using ellipse orientation
Mat rotationMatrix = Imgproc.getRotationMatrix2D(fittedEllipse.center, fittedEllipse.angle, 0.7);
Mat rotatedImage = new Mat();
Imgproc.warpAffine(imageMat, rotatedImage, rotationMatrix, imageMat.size(), Imgproc.INTER_LINEAR);

// Extract largest contour again from rotated image
List<MatOfPoint> contours2 = new ArrayList<>();
Imgproc.findContours(rotatedImage, contours2, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

double maxArea2 = 0;
MatOfPoint maxContour2 = new MatOfPoint();
int maxContour2Index = 0;

for (int i = 0; i < contours2.size(); i++) {
    double area = Imgproc.contourArea(contours2.get(i));
    if (area > maxArea2) {
        maxArea2 = area;
        maxContour2 = contours2.get(i);
        maxContour2Index = i;
    }
}

// Remove all other contours except largest one from both images
for (int i = 0; i < firstContoutcounter; i++) {
    if (i != (firstContourindicator - 1)) {
        Imgproc.drawContours(imageMat, contours, i, new Scalar(0, 0, 0), -1);
    }
}
Imgproc.drawContours(imageMat, contours, firstContourindicator - 1, new Scalar(255, 255, 255), -1);

for (int i = 0; i < contours2.size(); i++) {
    if (i != maxContour2Index) {
        Imgproc.drawContours(rotatedImage, contours2, i, new Scalar(0, 0, 0), -1);
    }
}
Imgproc.drawContours(rotatedImage, contours2, maxContour2Index, new Scalar(255, 255, 255), -1);

// Prepare cropped image for asymmetry analysis
Mat assymetryCropped = rotatedImage.clone();

// Find centroid of rotated image's max contour
Moments mNew = Imgproc.moments(maxContour2);
Point newCentroid = new Point(mNew.get_m10() / mNew.get_m00(), mNew.get_m01() / mNew.get_m00());

int imgHeight = assymetryCropped.rows();
int imgWidth = assymetryCropped.cols();

// Containers for top-bottom asymmetry
Mat topHalf = new Mat();
Mat bottomHalf = new Mat();
Mat blankRow = Mat.zeros(1, 600, CvType.CV_8UC1);

// Extract upper half (above centroid)
for (int i = (int) newCentroid.y; i > 0; i--) {
    topHalf.push_back(assymetryCropped.row(i));
}

// Extract lower half (below centroid)
for (int i = (int) newCentroid.y; i < imgHeight; i++) {
    bottomHalf.push_back(assymetryCropped.row(i));
}

// Now compute left-right asymmetry by flipping image
Core.transpose(assymetryCropped, assymetryCropped);
Core.flip(assymetryCropped, assymetryCropped, +1);

// Extract contours again after flip + transpose
List<MatOfPoint> contours3 = new ArrayList<>();
Imgproc.findContours(assymetryCropped, contours3, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

double maxArea3 = 0;
MatOfPoint maxContour3 = new MatOfPoint();
int maxContour3Index = 0;

for (int i = 0; i < contours3.size(); i++) {
    double area = Imgproc.contourArea(contours3.get(i));
    if (area > maxArea3) {
        maxArea3 = area;
        maxContour3 = contours3.get(i);
        maxContour3Index = i;
    }
}

// Get centroid of flipped-transposed image
Moments againMNew = Imgproc.moments(maxContour3);
Point flippedCentroid = new Point(againMNew.get_m10() / againMNew.get_m00(), againMNew.get_m01() / againMNew.get_m00());

// Prepare left and right parts for asymmetry
Mat leftHalf = new Mat();
Mat rightHalf = new Mat();

for (int i = (int) flippedCentroid.y; i > 0; i--) {
    leftHalf.push_back(assymetryCropped.row(i));
}
for (int i = (int) flippedCentroid.y; i < imgWidth; i++) {
    rightHalf.push_back(assymetryCropped.row(i));
}

// Balance top vs bottom halves by padding smaller one
if (topHalf.rows() < bottomHalf.rows()) {
    int diff = bottomHalf.rows() - topHalf.rows();
    for (int i = 0; i < diff; i++) {
        topHalf.push_back(blankRow);
    }
} else {
    int diff = topHalf.rows() - bottomHalf.rows();
    for (int i = 0; i < diff; i++) {
        bottomHalf.push_back(blankRow);
    }
}


try {
    // XOR the images to compute asymmetry
    bitwise_xor(data, data2, data3);
    Moments m22 = moments(data3, true);
    double ss = m22.m00;

    // Vertical alignment: pad smaller matrix to match height
    if (y1new < y11new) {
        int diff = y11new - y1new;
        for (int k = 0; k < diff; k++) {
            data7.push_back(Forcopy.row(0));
        }
    } else {
        int diff = y1new - y11new;
        for (int k = 0; k < diff; k++) {
            data8.push_back(Forcopy.row(0));
        }
    }

    // XOR the vertical halves
    Mat data9 = new Mat();
    bitwise_xor(data7, data8, data9);
    Moments m21 = moments(data9, true);
    double ss1 = m21.m00;

    // Final asymmetry score
    double avg = (ss + ss1) / (2 * mnew.m00);

    // Color detection counters
    int black1 = 0, white1 = 0, red1 = 0, light1 = 0, dark1 = 0, bluegray = 0;
    int colorAreaCounter = 0;

    for (int iu = 0; iu < imageMat2.rows(); iu++) {
        for (int ju = 0; ju < imageMat2.cols(); ju++) {
            if (imageMat.get(iu, ju)[0] == 255) {
                colorAreaCounter++;

                double r = imageMat2.get(iu, ju)[0];
                double g = imageMat2.get(iu, ju)[1];
                double b = imageMat2.get(iu, ju)[2];

                if (r <= 62 && g <= 52 && b <= 52) black1++;
                else if (r >= 205 && g >= 205 && b >= 205) white1++;
                else if (r >= 150 && g < 52 && b < 52) red1++;
                else if (r >= 150 && r <= 240 && g > 50 && g <= 150 && b <= 100) light1++;
                else if (r > 62 && r < 150 && g < 100 && b < 100) dark1++;
                else if (r <= 150 && g >= 100 && g <= 125 && b >= 125 && b <= 150) bluegray++;
            }
        }
    }

    // Count unique color zones (threshold: >0.1% of region)
    int colors = 0;
    double rc = colorAreaCounter;
    if (red1 / rc > 0.001) colors++;
    if (bluegray / rc > 0.001) colors++;
    if (black1 / rc > 0.001) colors++;
    if (white1 / rc > 0.001) colors++;
    if (dark1 / rc > 0.001) colors++;
    if (light1 / rc > 0.001) colors++;

    // Mean and standard deviation of RGB channels within lesion
    MatOfDouble mean = new MatOfDouble();
    MatOfDouble stddev = new MatOfDouble();
    Core.meanStdDev(imageMat2, mean, stddev, imageMat);

    double rMean = mean.get(0, 0)[0];
    double gMean = mean.get(1, 0)[0];
    double bMean = mean.get(2, 0)[0];
    double meanVal = (rMean + gMean + bMean) / 3;

    double rStd = stddev.get(0, 0)[0];
    double gStd = stddev.get(1, 0)[0];
    double bStd = stddev.get(2, 0)[0];
    double stdVal = (rStd + gStd + bStd) / 3;

    // Draw results on image
    Bitmap resizedBitmap = Bitmap.createScaledBitmap(newbitmap, imageMat2.cols(), imageMat2.rows(), false);
    Imgproc.drawContours(imageMat2, contours, firstContourindicator - 1, new Scalar(255, 255, 255), 3);
    if (max_contour.toArray().length >= 5) {
        Imgproc.ellipse(imageMat2, newellipse, new Scalar(0, 255, 0), 2);
    }
    Utils.matToBitmap(imageMat2, resizedBitmap);
    imageView.setImageBitmap(resizedBitmap);

    // Feature engineering
    double IRA = epsilon / maxArea;
    double IRB = epsilon / major;
    double cd = 1.0 / minor;
    double is = 1.0 / major;
    double sd = cd - is;
    double IRC = epsilon * sd;
    double IRD = major - minor;

    // Model weights
    double w1 = -0.0358326452, w2 = 0.278801975, w3 = -0.134581583;
    double w4 = 0.00295291939, w5 = -1.78501516, w6 = 6.23128637;
    double w7 = -0.382788048, w8 = 1.5647672, w9 = -0.0168759429;
    double w10 = 0.142887996;
    double bias = -1.48299157;

    // Model input variables
    float newira = (float) IRA;
    float newirb = (float) IRB;
    float newirc = (float) IRC;
    float newird = (float) IRD;
    float newb = (float) B;
    float newavg = (float) avg;
    float newmeanval = (float) meanVal;
    float newstdval = (float) stdVal;
    float newEdgeAbruptness = (float) edgeabruptness;

    // Linear model for classification
    float score = (float) (newira * w1 + newirb * w2 + newirc * w3 + newird * w4
            + newb * w5 + newavg * w6 + colors * w7 + newEdgeAbruptness * w8
            + newmeanval * w9 + newstdval * w10 + bias);

    String result = (score > 0) ? "Malignant Melanoma" : "Benign Melanoma";

    // Show result on screen
    textview1.refreshDrawableState();
    textview1.setText(
            "IRA: " + newira +
            "\nIRB: " + newirb +
            "\nIRC: " + newirc +
            "\nIRD: " + newird +
            "\nCI: " + newb +
            "\nColors: " + colors +
            "\nAsymmetry: " + newavg +
            "\nMean: " + newmeanval +
            "\nStandard Deviation: " + newstdval +
            "\nEdge Abruptness: " + newEdgeAbruptness +
            "\nScore: " + score +
            "\nClassification Result: " + result
    );
} catch (Exception e) {
    textview1.setText("Image cannot be processed");
}



