# Fashion Silhouette Analysis

This project explores how the overall shape of clothing in older fashion images compares to modern fashion. The goal was to use simple Digital Image Processing techniques learned in class to extract silhouettes from images and analyze their shape, without relying on complex machine learning methods.

The project focuses on turning fashion images into clean silhouettes, measuring basic width information, and visualizing image features to better understand differences in clothing structure over time.

---

## What This Project Does

- Converts color images to grayscale
- Enhances contrast to improve older or faded images
- Extracts clean silhouettes using thresholding and morphology
- Removes background noise by keeping only the main subject
- Measures silhouette width at three heights (25%, 50%, and 75%)
- Draws measurement lines directly on the silhouette
- Generates Histogram of Oriented Gradients (HOG) visualizations

---

## Why I Built This

I was interested in how fashion silhouettes have changed over time, especially when looking at older black-and-white photos compared to modern images. Instead of using advanced models, I wanted to see how much information could be captured using basic image processing tools.

This project shows that even simple techniques like thresholding, morphology, and connected components can be useful for analyzing visual patterns.

---

## How It Works 

First, each image is converted to grayscale and enhanced to improve contrast. The image is then thresholded to separate the clothing from the background. Morphological operations are applied to clean up the result, and connected components are used to isolate the main silhouette.

Once the silhouette is extracted, its width is measured at three different heights to get a basic sense of shape. A separate script computes and saves HOG feature visualizations to show texture and edge structure.

---

## Tools Used

- Python
- OpenCV
- NumPy
- scikit-image

---

## Limitations and Future Improvements

- The method works best when there is strong contrast between the subject and background
- Results can be affected by lighting or busy backgrounds
- Measurements are not normalized across image sizes

In the future, this project could be improved by:
- Using adaptive thresholding methods
- Normalizing silhouettes before measuring width
- Comparing HOG features quantitatively
- Expanding the dataset with more labeled images

---

## Final Notes

This project was completed as part of a Digital Image Processing course and reflects a hands-on approach to understanding how classic computer vision techniques can be applied to real-world visual data.

