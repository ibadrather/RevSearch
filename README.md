# RevSearch

RevSearch is a Minimum Viable Product (MVP) showcasing a car reverse image search application. It leverages cutting-edge machine learning, computer vision, and cloud-based technologies to provide an efficient and accurate image search experience. Built as a self-initiative, this project demonstrates end-to-end machine learning workflow expertise.

**Note:** The deployment has been taken down due to running costs.

---

## Workflow

1. **Image Upload**: Users upload an image in various formats (JPG, PNG, BMP, etc.).
2. **Feature Encoding**: The uploaded image is encoded into a feature vector using a trained neural network encoder (EfficientNet).
3. **Similarity Search**: The encoded feature vector is compared to precomputed feature vectors in the database.
4. **Top Matches**: The system retrieves the top 10 image URLs based on similarity scores.
5. **Image Retrieval**: Top 10 images are fetched from AWS S3 storage via AWS API Gateway and AWS Lambda.

> **Note**: RevSearch is currently not deployed due to associated operational costs.

---

## Project Details

- **Company**: Self-initiative project for applying end-to-end machine learning workflows.
- **Timeline**: April 2022 - May 2022
- **Codebase**:
  - Backend, Frontend, and Core Technology
    - [DeepImageSearchAPI](https://github.com/ibadrather/DeepImageSearchAPI)
    - [RevSearch](https://github.com/ibadrather/RevSearch)

---

## Key Features

- Upload images in various formats (JPG, PNG, BMP, etc.) for reverse search.
- Interactive slider to select up to 6 similar images.
- Powered by the EfficientNet neural network architecture for accurate feature extraction.
- Responsive and seamless user experience powered by FastAPI.
- Fully interactive **Reverse Image Search WebUI**.

---

## Technologies Used

| **Category**         | **Technologies**                                                      |
|-----------------------|----------------------------------------------------------------------|
| Core Technologies     | Python, PyTorch, ONNX, ONNX Runtime, Pandas                         |
| Data Preprocessing    | Albumentations                                                      |
| Model Optimization    | MLflow, Optuna                                                      |
| Web App & Deployment  | Streamlit, FastAPI, Docker, Heroku                                  |
| Cloud Services        | AWS S3, AWS API Gateway, AWS Lambda                                 |
| CI/CD & Code Quality  | GitHub Actions, Black, Pytest                                       |
| Image Processing      | PIL                                                                 |

---

## About the Dataset

- **Source**: Stanford University AI Lab’s Cars dataset.
- **Composition**:  
  - 16,185 images across 196 car classes.  
  - Serves as the foundation for the feature extractor (encoder).

---

## DeepSearchLite Integration

RevSearch integrates **DeepSearchLite**, a custom lightweight library, for fast and efficient similarity searches with minimal dependencies.  
Find it on PyPi: [DeepSearchLite](https://pypi.org/project/DeepSearchLite/)

---

## Challenges and Future Improvements

1. **Dataset Limitations**:
   - Cars dataset (16,185 images) is outdated, impacting accuracy for newer models.
2. **MVP Status**:
   - Currently demonstrates potential but requires further enhancements.
3. **Next Steps**:
   - Expand the dataset to include more images and newer models.
   - Refine search algorithms for improved accuracy and speed.
   - Incorporate user feedback for additional features and functionality.

---

This project highlights expertise in advanced technologies and practical solutions for the automotive domain.  
