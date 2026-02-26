# **Technical Roadmap: Reject Inference Methodological Comparison**

This roadmap details the technical steps for implementing a hybrid model based on latent representations (Custom Deep Learning) versus a classical baseline model, specifically tailored for the Nigerian Credit Risk dataset (e.g., Kowope Mart).

## **Step 1: Data Preprocessing (Advanced)**

This stage establishes the foundation for a fair comparison between the two routes.

**Reference Implementation:** The EDA, initial cleaning, and feature engineering pipeline will be primarily based on the **aifenaike/DSN\_KOWOPE** repository (1st Place, DSN 2020), ensuring best-in-class handling of local data nuances.

### **1.1. Initial Cleaning & Standardization (Shared Pipeline)**

* **LGA Correction:** leveraging aifenaike's mapping strategies to correct spelling errors in LGA (Local Government Area) and map them to the standard 774 Nigerian LGAs (e.g., standardizing "Suru-lere" vs. "Surulere").  
* **Currency & Format Parsing:** Cleaning Loan\_Amount and Total\_Income by stripping currency symbols ('₦'), commas, and whitespace before casting to float.  
* **Category Consolidation:** Merging redundant banking categories (e.g., combining Savings (Tier 1\) and Savings) based on domain logic found in the reference solution.  
* **De-duplication:** Removing duplicate applications to ensure a strict train/test separation.

### **1.2. Feature Engineering (Shared Pipeline)**

* **Geo-Political Zoning:** Aggregating LGAs into Nigeria's 6 Geopolitical Zones (North-West, South-East, etc.) to capture macro-economic risk factors.  
* **Temporal Features:** Converting raw dates into duration features (e.g., Time\_since\_last\_loan, Age\_of\_Account).  
* **Interaction Ratios:** creating Debt-to-Income (DTI) ratios if applicable.

### **1.3. Data Splitting strategy**

* **Group ![][image1]:** Financed applicants (includes the Target label Good\_Bad).  
* **Group ![][image2]:** Non-financed/rejected applicants (unlabeled).  
* **Hold-out Set:** A strict hold-out set (20%) is reserved from ![][image1] for final evaluation.

### **1.4. The Preprocessing Fork (Handling Missing Values)**

Here the pipeline diverges to satisfy the distinct requirements of Deep Learning (Route A) and Tree-based models (Route B).

* **Route A (Hybrid / VIME Preparation):**  
  * **Missing Values:** Impute with a distinct placeholder (e.g., -1) **AND** generate a corresponding binary `is_missing` indicator. This critical step allows the VIME Mask Estimator to learn the specific "missingness" patterns associated with informal employment while ensuring valid numerical input for the Encoder.
  * **Indicators:** Ensure binary mask columns are generated for key variables.  
  * **Scaling:** **Mandatory** application of StandardScaler or MinMaxScaler on all numerical inputs to ensure neural network convergence.  
  * **Encoding:** Use Entity Embeddings or Target Encoding (scaled) for high-cardinality variables like LGA.  
* **Route B (Classical Baseline Preparation):**  
  * **Missing Values:** Follow the aifenaike baseline strategy (e.g., filling with \-999) OR apply **MICE** (Multiple Imputation) to create a complete rectangular dataset required for standard baseline comparison.  
  * **Scaling:** No scaling required (Tree-based models are invariant to monotonic transformations).  
  * **Encoding:** Use Target Encoding (with smoothing) as per the reference solution.

## **Step 2: Route A \- Deep Learning Implementation (VIME)**

This phase focuses on the custom development of the neural network architecture:

1. **Network Architecture:** Development of an Encoder and two specialized heads:  
   * **Mask Estimator:** A network designed to predict the binary mask vector of the corrupted/hidden inputs.  
   * **Feature Reconstructor:** A network designed to reconstruct the original raw feature values (performing internal imputation).  
2. **Training Process (Self-Supervised):** Training on the full population (![][image3]) using a composite loss function (**Mask Cross-Entropy** + **Feature Reconstruction MSE**) to learn latent representations of both accepted and rejected applicants.  
3. **Embedding Extraction:** Generating latent vectors (embeddings) for every applicant in the dataset.

## **Step 3: Route A \- Hybrid Reject Inference Process**

1. **Teacher Model:** Training a CatBoost model on ![][image1] using a combination of **raw tabular features** AND the **embeddings** generated in Step 2\.  
2. **Pseudo-Labeling:** Predicting default probabilities for the ![][image2] group.  
3. **Filtering (Inlier Detection):** Utilizing an **Isolation Forest** algorithm on the embeddings to select only those rejected applicants who are statistically similar (inliers) to the financed population. **Critical Step:** This mitigates covariate shift by discarding rejected applicants that are too distinct from the training distribution.
4. **Final Model Training:** Training the final discriminative model on the unified dataset (![][image1] \+ selected high-confidence ![][image2] samples with pseudo-labels).

## **Step 4: Route B \- Classical Approach (Baseline)**

1. **Label Generation:** Training a standard CatBoost model on ![][image1] (processed via the Route B pipeline: MICE/-999, no embeddings).  
2. **Pseudo-Labeling:** Predicting labels for the ![][image2] group based on tabular data alone.  
3. **Final Model Training:** Training the final model on the merged classical dataset (![][image1] \+ ![][image2] pseudo-labeled) without deep learning components.

## **Step 5: Comparison and Evaluation**

1. **Performance Metrics:** Comparison of **AUC-ROC** and **Precision-Recall** on the Hold-out set (![][image1] test set).  
2. **Edge Case Analysis:** Utilization of the **AUK (Area Under the Kickout)** metric to assess performance gains specifically among the "grey area" applicants.  
3. **SHAP Analysis:** Investigating the impact of embeddings on the hybrid model's decisions versus the classical model's feature importance (e.g., seeing if the model relies less on Total\_Income and more on behavioral embeddings).
4. **Stability Metrics:** Calculating the **PSI (Population Stability Index)** between the accepted population and the pseudo-labeled rejected population to ensure the model isn't generating wildly different score distributions.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAYCAYAAAD6S912AAABh0lEQVR4Xu2UvUsDQRDFLySIoBYi8ZD7voCBFFqInSCIhf+AlaWglYU2FtZiLykFQVHwA+wttBNE0NJKEBs7C9FKMP5G98JmY7wLxs4Hj7t782Z2mdlby/pHRxAEwRS8833/ISOnzRo6chSrhmF4AEP5FhFtC74jzShfnvdJtHvP88br2SaiKLIxHZZKpcFEYwf9aFeS7Lquk+jFYrEXbddxHDfRmiDbZ+VlXWORURKf4TGfhURXC22Wy+U+zd4ITLNxHA/rGklzsEZsVdfZ2QDagqXakhmqf28kT5ixttGqf98gL14rbbeYxij2avZPhwyQ+BHeS57zZrwBrfqnQ3n24RoDHDHjOnIU2k7rn/SYk7Fo6k1I658cfGJV+MjrKc+VSqXSZfrqyNI/YkPwAm9sxj4hZxDDNXyCNY0v8FYW0f3y26GfqQn/HsHXQPastOOSFRRb/+kEtAUG0U3BEz/l+soEuWnUlM9hZMbbglwKFLmBO+xuw+pA/woyDIot2bbdYwb/FB+ysWeJgJ6n2wAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAACC0lEQVR4Xu2UTWtTQRSGb0gK9aN+UNMrucn9SKLZCC4uXRTFQltECu50lb2upFjB/oBS6EYqwVURRARBEVooBNRuuitU3LaLFsQf0EWpK8H4HJhph0kbmnsNiPjCS+5558w5M+dMjuP8xz+JIAjG4I7v+99PyAk7RrfIkPBFGIbvYCi2iGgv4S+kO8ovy/co2rdSqTR8sDsJoihyCfS+UqkMaY2bXETbkATFYtHTej6fP4v2xvO8otYSQUrFDR6bGge5TvA9+AEzp3V1mEatVhsw3LsHge6Xy+WrpkbgOmyxNmPq3HAQ7YGjWvBHofr5kwQ37bWe4Lh+JkBGYvGbtRfagGNMwh92P7sBhz3F/kW4Cp/Z6204rp/dQNpCjBWJwaO8ba/bkJK8SttP9k/Dhq0fiU79RIvgW3zuwRtqmDzn5Z/XPtgX4Dx+W/ALnK1Wq+fMOG3o0E+pwCMpFWvbcEREErxGv2v46QHS7Fgp+Y/i9BXuwpbBfbgpB8Etx82vkOShlB87I8GxP/nWHJZpxb519LKpJ4Xud10M5u817LVCoXDJdJIbwo+pp5ZA9fuzurnYM9hz/I5T9lvaTw4lZT/cmQJqHjf1w+H7CQkX4FQcx33aD70hbTjcmQ5ZSnnaFFzXPeOoOUyifniZQywH6qH1FJKQREsq4aJMJNunJyDZJEmfym3ttb8CvwG+E4TfSdlsHAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEYAAAAYCAYAAABHqosDAAADTklEQVR4Xu2YTWgTURDHt6SK3x9ojTZpNh/VQhE9BAWxKGgRFQTB9tSjoCAUUVEPghcp6kGU2lMpFC0W/AAFoeAHigiiKHrUg4J48eZB9CRYf0Pe2pdpsrtJs7FC/vCn+2ZmZ+fNm3nvpY7TQAMNNBAxXNfdAT+lUqkvIdmtfdQD9Y6ziY8NpdPpmzAtYxEiG4G/Ee02djGetyP73NbWtunv2yGQSCSSra2tK7VcI5PJuMlkcr6WG0QeZxEIJo6TW7lcbpUnI9PLkb0W5wSa8OQtLS2LkI3JRD1ZGODvNNyn5Rr4vopdXssF9YizCFJuZPiYLSOIjTj+Du8wbPbkJpDBjo6OxZZ5IGqRmHrEWQSc9Gaz2XW2DKd9cFImZMtZgRXIDjmmjMOiRomJPM5AuIW+/YXzLq2rBrVITCnUOk5fmDKc1rclEBNbJ8SqRJGYCuIMQpOZR0wriiCB8bGfrupbG7IBor+N7Sv+HtR6jYgSExhnEOQE5P1h+Bhe0voiuGX61oaxGYdn2AA3aL1GFIkJE2cQeLcLH/fFB/PYpfU2pKxG3YC+RT/CCXFYy8sB++MhEzMQ8ogNFWcQJC44qOXTkAroW7lYoRuCX3l8KI47OzvnajsN/HbDs47PfiQnCf7GwlwE/eJEloHj2PTAreZCeIUTbalnw3gZvIDdB/gGnmtvb19i+ymClLEb0Lfo1sAX2Ga1rhzy+fwcCY53ep0SyYnH4wvdwglzQOtKwSdOqaR+aQt0H+EWEfLta7pizSVwomzFyd0Ag7fwG5y0+AO+lyBse7lmI38iq2bLg2A2upPwOTxiVrSHoC8yfgn3OCWS5iFknM18Z620OeNRxk2SAKnulPrdJC0r361kgX3hFja9G47PJPxAkPNY0c0mKftJdM4JOiorg7f/9MkA/+sZP9MtKpUCH8zodmyDDw6kZnASRA2pZGJ85FW6xGpi3smCbPPsJHHSYlNvzgCy2ji8q8tyNiFT+P004W22PJ8g3svwqOx1nh3yQWm5qTerhOlVOZWewozWzyLEaJsFtkA2d8e0viwuXE2i7rlmc64a5ih9B6/j8LxT5f7yr2FVvCRl2Od/P6HRLD2Jw36T/f8WzGMv8zglVaN1DYTAH/1qFxO1eKQrAAAAAElFTkSuQmCC>