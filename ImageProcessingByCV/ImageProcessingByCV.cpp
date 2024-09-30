#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mpi.h>


using namespace cv;

cv::Mat readImage(const std::string& filePath) {
    // Open the image file in binary mode
    std::ifstream file(filePath, std::ios::binary);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open image file" << std::endl;
        return cv::Mat();
    }

    // Read the contents of the file into a vector
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});

    // Create a cv::Mat object from the buffer
    cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);

    // Check if the image was loaded successfully
    if (image.empty()) {
        std::cerr << "Error: Unable to decode image" << std::endl;
        return cv::Mat();
    }

    return image;
}

// 1-Create a separate image for grayScal
void convertToGrayscale(cv::Mat& image) {
    double start_time = cv::getTickCount(); // Start
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "\n\nGray operation completed successfully : " << execution_time << " seconds" << std::endl;
}


// 2-Create a separate image for Gaussian Blur 
void applyGaussianBlur(cv::Mat& image) {
    double start_time = cv::getTickCount(); // Start
    cv::Mat blurredImg;
    cv::GaussianBlur(image, blurredImg, cv::Size(5, 5), 3);
    blurredImg.copyTo(image);
    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "\n\nGaussian operation completed successfully : " << execution_time << " seconds" << std::endl;
}


// 3-Create a separate image for edge detection
cv::Mat edgeDetectionImage(cv::Mat& image) {
    cv::Mat edge_image;
    Canny(image, edge_image, 100, 200);
    image = edge_image.clone();
    return edge_image;
}
void edgeDetection(cv::Mat& local_image) {
    double start_time = cv::getTickCount(); // Start
    local_image = edgeDetectionImage(local_image);
    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "\n\nEdge Detection operation completed successfully : " << execution_time << " seconds" << std::endl;
}

// Create a separate image for rotateImage
cv::Mat rotateImage(const cv::Mat& image, double angle) {
    cv::Mat rotatedImg;
    cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(image, rotatedImg, rotationMatrix, image.size());
    return rotatedImg;
}

// 4-Rotate the local chunk of the image
void rotateLocalImage(cv::Mat& local_image, double angle) {
    double start_time = cv::getTickCount(); // Start

    local_image = rotateImage(local_image, angle);

    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();

    std::cout << "\n\nRotate Image  operation completed successfully: " << execution_time << " seconds" << std::endl;
}


// 5-Function to scale the local chunk of the image
void scaleImage(cv::Mat& image, double scale_factor) {
    double start_time = cv::getTickCount(); // Start
    cv::Mat scaledImg;
    cv::resize(image, scaledImg, cv::Size(), scale_factor, scale_factor);
    scaledImg.copyTo(image);
    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "\n\nScale Image  operation completed successfully: " << execution_time << " seconds" << std::endl;
}

// 6-Function to apply a custom filter kernel to the local chunk of the image
void applyCustomFilter(cv::Mat& image, cv::Mat& kernel) {
    double start_time = cv::getTickCount(); // Start
    cv::Mat filteredImg;
    cv::filter2D(image, filteredImg, -1, kernel);
    filteredImg.copyTo(image);
    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "\n\napply Custom Filter operation completed successfully: " << execution_time << " seconds" << std::endl;
}


// 7-Function to perform histogram equalization on the local chunk of the image
void equalizeHistogram(cv::Mat& image) {
    double start_time = cv::getTickCount(); // Start
    cv::Mat equalizedImg;
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY); // Convert image to grayscale
    cv::equalizeHist(image, equalizedImg);
    equalizedImg.copyTo(image);
    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "\n\nequalize Histogram operation completed successfully : " << execution_time << " seconds" << std::endl;
}

// 8-Function to convert the local chunk of the image to a different color space
void convertColorSpace(cv::Mat& image, int conversion_code) {
    double start_time = cv::getTickCount(); // Start
    cv::cvtColor(image, image, conversion_code);
    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "\n\nColor space operation completed successfully : " << execution_time << " seconds" << std::endl;
}


// 9-Function to perform global thresholding
void globalThresholding(cv::Mat& image, int threshold) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat thresholdedImage;
    cv::threshold(grayImage, thresholdedImage, threshold, 255, cv::THRESH_BINARY);

    thresholdedImage.copyTo(image);
}




// 10-Function to perform local thresholding
void localThresholding(cv::Mat& image, int blockSize, double C) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat thresholdedImage;
    cv::adaptiveThreshold(grayImage, thresholdedImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, blockSize, C);

    thresholdedImage.copyTo(image);
}



// 11-Function to apply median filter to the local chunk of the image
void applyMedianFilter(cv::Mat& image, int kernelSize) {
    double start_time = cv::getTickCount(); // Start
    if (kernelSize % 2 == 0) { // Check if the kernel size is even
        std::cout << "Kernel size must be an odd integer. Setting kernel size to 3." << std::endl;
        kernelSize = 3; // Set the kernel size to 3 if it's even
    }
    cv::Mat filteredImage;
    cv::medianBlur(image, filteredImage, kernelSize);
    filteredImage.copyTo(image);
    double end_time = cv::getTickCount(); // End
    double execution_time = (end_time - start_time) / cv::getTickFrequency();
    std::cout << "\n\nMedian filter operation completed successfully : " << execution_time << " seconds" << std::endl;
}



// Function to measure the execution time of a function
//void measureExecutionTime(void (*function)(cv::Mat&), cv::Mat& image) {
//    double startTickCount = cv::getTickCount(); // Get the initial tick count
//
//    // Call the function to measure execution time
//    function(image);
//
//    double endTickCount = cv::getTickCount(); // Get the final tick count
//    double elapsedTime = (endTickCount - startTickCount) / cv::getTickFrequency(); // Calculate elapsed time in seconds
//
//    std::cout << "Image Processing operation completed successfully in  " << elapsedTime << " seconds" << std::endl;
//}



// Define a custom filter kernel 
cv::Mat customKernel = (cv::Mat_<float>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);

// Define a apply Closing structure
cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

// Define a segmented Image
cv::Mat segmentedImage;

cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//select function to apply filter
void applyImageProcessing(cv::Mat& image, int selectedFunction) {

    switch (selectedFunction) {
    case 1:
        //std::cout << "\n\nYou have selected convert To Grayscale." << std::endl;
        //convertToGrayscale(image);
        ////measureExecutionTime(convertToGrayscale, image);

        break;
    case 2:
        std::cout << "\n\nYou have selected Gaussian Blur." << std::endl;
        applyGaussianBlur(image);
        // measureExecutionTime(applyGaussianBlur, image);
        break;
    case 3:
        std::cout << "\n\nYou have selected edge Detection." << std::endl;
        edgeDetection(image);
        // measureExecutionTime(edgeDetection, image);
        break;
    case 4:
        float an;
        std::cout << "\n\nYou have selected rotate Image." << std::endl;
        std::cout << "\n\nPlease Enter an angle to rotate." << std::endl;
        std::cin >> an;
        rotateLocalImage(image, an);

        break;
    case 5:
        float scal;
        std::cout << "\n\nYou have selected scale Image" << std::endl;
        std::cout << "\n\nPlease Enter an Scale Factor ." << std::endl;
        std::cin >> scal;
        scaleImage(image, scal);

        break;
    case 6:
        std::cout << "\n\nYou have selected apply Custom Filter" << std::endl;
        applyCustomFilter(image, customKernel);

        break;
    case 7:
        std::cout << "\n\nYou have selected scale Image" << std::endl;
        equalizeHistogram(image);
        /* measureExecutionTime(equalizeHistogram, image);*/
        break;
    case 8:
        std::cout << "\n\nYou have selected convert Color Space" << std::endl;
        convertColorSpace(image, cv::COLOR_BGR2HSV);

        break;
    case 9:
        std::cout << "\n\nYou have selected global Thresholding" << std::endl;

        globalThresholding(image, 3);
        break;
    case 10:
        std::cout << "\n\nYou have selected local Thresholding" << std::endl;
        double mean, stdDev;
        localThresholding(image, 7, 9);
        break;
    case 11:
        std::cout << "\n\nYou have Median Filter segmenta Image" << std::endl;
        int ks;
        std::cout << "Enter kernal Size";
        std::cin >> ks;
        applyMedianFilter(image, ks);

        break;
    default:
        std::cout << "\n\nInvalid function selection" << std::endl;
        break;
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int sel;
    std::cout << "Welcome to Parallel Image Processing with MPI\n\n\nPlease choose an image processing operation :\n\n1.convert To Grayscale\n2.Gaussian Blur\n3.Edge Detection\n4.Image Rotation\n5.Image Scaling\n6.Color Space Conversion\n7.Histogram Equalization\n8.Image Filtering(Custom Filters)\n9.Global Thresholding\n10.Local Thresholding\n11.Median Filter\n\n" << std::endl;
    std::cout << "\nEnter your choice (1-10):";
    std::cin >> sel;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::string imagePath;
    std::string outPath;
    cv::Mat image;
    cv::Mat outImage;

    // Rank 0 loads the image and prompts for input
    if (rank == 0) {
        std::cout << "\n\nPlease enter the filename of the input image (e.g., input.jpg): ";
        std::cin >> imagePath;
        image = cv::imread(imagePath);

        std::cout << "\n\nPlease enter the filename for the output blurred image (e.g., output.jpg): ";
        std::cin >> outPath;

        if (image.empty()) {
            std::cout << "Error: Unable to load image" << std::endl;
            MPI_Finalize();
            return -1;
        }
    }

    // Broadcast image dimensions to all processes
    int rows, cols;
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide rows among processes
    int rows_per_process = rows / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? rows : start_row + rows_per_process;

    // Allocate memory for the local chunk of the image
    cv::Mat local_image(end_row - start_row, cols, CV_8UC3);

    // Scatter image data to all processes
    MPI_Scatter(image.data, local_image.rows * cols * 3 * sizeof(unsigned char), MPI_BYTE,
        local_image.data, local_image.rows * cols * 3 * sizeof(unsigned char), MPI_BYTE,
        0, MPI_COMM_WORLD);

    // Process the local chunk of the image
    // Here you can perform any desired image processing operations on the local_image

    // Convert local chunk to grayscale
    applyImageProcessing(local_image, sel);
    if (rank == 0)
        cv::imwrite("mp.jpg", local_image);
    std::cout << "\n\nProcessing image ..." << std::endl;
    // Display the original and blurred images
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Image", image);

    cv::namedWindow("Processed Image", cv::WINDOW_NORMAL | cv::WINDOW_AUTOSIZE);
    cv::imshow("Processed Image", local_image);
    // Wait for a key press
    cv::waitKey(0);
    if (rank == 1)
        cv::imwrite("outPath1.jpg", local_image);
    if (rank == 2)
        cv::imwrite("outPath2.jpg", local_image);


    // Gather processed data from all processes
    MPI_Gather(local_image.data, local_image.rows * cols * 3 * sizeof(unsigned char), MPI_BYTE,
        image.data, local_image.rows * cols * 3 * sizeof(unsigned char), MPI_BYTE,
        0, MPI_COMM_WORLD);

    // Rank 0 saves the result
    if (rank == 0) {
        std::cout << "\n\nBlurred image saved as " << outPath << std::endl;
        local_image = cv::imwrite(outPath, image);
    }

    std::cout << "\n\nThank you for using Parallel Image Processing with MPI." << outPath << std::endl;

    MPI_Finalize();

    return 0;
}