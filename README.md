# 👤 find-ideal-face

**find-ideal-face** is a fun project for me to find my ideal facial features. In the code, I combined faces of Tsuyu and Karina to find a blending of the features to see if it creates ideal features.
The project practices fundamental face morphing techniques such as Delaunay triangulation and Affline transformation to generate a well-blended image of 2 input images based on alpha parameter.

---

## 🚀 Overview
Averaging faces isn't as simple as lowering the opacity of multiple layers. Because every human has different proportions, features like the eyes and mouth will never align naturally. 

This project solves that by:
1.  **Extracting** facial geometry (68 landmarks).
2.  **Calculating** a "Mean Shape" from all inputs.
3.  **Warping** every individual face to fit that Mean Shape.
4.  **Blending** the warped images to produce a crisp, averaged result.

---

## 📐 Technical Deep Dive

The core of this project relies on two mathematical concepts: **Delaunay Triangulation** and **Affine Transformations**.

### 1. Delaunay Triangulation
**What is it?** Delaunay Triangulation is an algorithm that connects a set of points (facial landmarks) into a mesh of non-overlapping triangles. It is designed so that the minimum angle of all the angles in the triangles is maximized, avoiding "sliver" triangles.

**Why we need it:** We start with facial landmarks, however, we cannot just apply affline transformation on landmark points, transform a certain region of the face. To do so, we need a mesh -- some kind of face "mask" divided into certain geometrical pieces. The best geometry here is a triangle, as we can divide a mask into pieces without any gaps. Also, finding transformation mappings of 68 landmarks would take too long and would be inaccurate.

### 2. Warp Affine (Affine Transformation)
**What is it?** An Affine Transformation is a geometric transformation that preserves points, straight lines, and parallelism. It allows for rotation, scaling, and translation. In this project, we use it to transform the pixels inside each Delaunay triangle from their original positions to their new positions in the "average" face.

**Why we need it:** Once we have our triangles, the triangles in "Face A" will be slightly different shapes than the triangles in the "Average Face." 
* **Matrix Calculation:** We calculate an Affine Transform Matrix for every corresponding triangle pair.
* **Pixel Mapping:** We use `cv2.warpAffine` to "stretch" the pixels from the source triangle into the target triangle shape.
* **Seamlessness:** This process allows us to align features with sub-pixel accuracy, which is what makes the final averaged face look like a real person rather than a blurry smudge.

---

## Challenges: 
* Difficult intuition / mathematical concepts
* Was going to use mediapipe, but had too many landmarks. I had to convert 400~ mediapipe landmarks into 68 dlib landmarks through another script to avoid jagged/weird morphings.
