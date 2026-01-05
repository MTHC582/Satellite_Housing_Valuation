# Satellite Housing Valuation: The "ENGINE + BRAIN" Approach

### _A Multimodal Deep Learning Pipeline for Real Estate_

> **"Combining the raw calculation power of numbers (ENGINE) with the visual intuition of satellite imagery (BRAIN)."**

This project implements a **Multimodal Neural Network** that fuses traditional tabular data (bedrooms, sqft, location) with high-resolution **Satellite Imagery** to predict property values. By "looking" at the neighborhood context (greenery, density, road proximity) via a CNN, the model captures value drivers that spreadsheets alone miss.

---

## Project Overview

**The Challenge:** Traditional valuation models relying solely on spreadsheets (Tabular Data) fail to capture "Curb Appeal" or "Neighborhood Vibes."
**The Solution:**

- **The ENGINE (Tabular Branch):** A Feed-Forward Network (FFN) processes exact housing specs (bed, bath, sqft) using `pandas` and `openpyxl`.
- **The BRAIN (Visual Branch):** A ResNet-based CNN processes satellite images to extract environmental features.
- **The Data:** A custom dataset of **22,000 properties**, with imagery programmatically fetched via the **Mapbox Static API**. (Took about 2.3 GB of DATA,as per the mentioned tools)

---

## NOTE:

I used **Python 3.11** as it ensures compatibility with the specific CUDA-enabled PyTorch wheels used in this project. To avoid dependency conflicts, I strongly suggest using the same version.

---

## 📂 Repository Structure

```text
Satellite_Housing_Valuation_DemoModel/
│
├── README.md               
├── requirements.txt       
├── src/
│   ├── data_fetcher.py     
│   ├── dataset.py         
│   ├── models.py           
│   ├── train.py            
│   └── predict.py         
│
├── data/
│   ├── train(1).xlsx     
|   ├── test2.xlsx
│   └── images/            
│
└── best_model.pth      
```

---

## Source Code Guide

### data_fetcher.py:

- The Collector.
- Takes the Excel file with Latitude/Longitude coordinates.
- Connects to the Mapbox Static API (or Google Maps).
- Downloads a satellite snapshot for each property and saves it as ID.jpg in the data/images folder.

### dataset.py:

- The Data Pipeline.
- Loads the Excel file using pandas.
- Handles missing values (fills with 0 or mean).
- Locates the corresponding image in data/images/ based on the Property ID.
- Applies image transformations (Resize to 224x224, Convert to Tensor).

### models.py:

- The Brain & Engine.
- Defines the ValuationModel class.
- Image Branch: Uses a ResNet-style Convolutional Neural Network (CNN) to extract visual features.
- Tabular Branch: Uses Fully Connected Layers (Linear -> ReLU) to process numerical data.
- Fusion: Concatenates both outputs and passes them through a final regression layer to predict price.

### train.py:

- The Teacher.
- Sets up the training loop (Forward Pass -> Loss Calculation -> Backpropagation).
- Uses MSELoss (Mean Squared Error) to measure accuracy.
- Optimizes weights using the Adam optimizer.
- Saves the model with the lowest validation loss as best_model.pth.

### predict.py:

- The Valuator.
- Loads the trained best_model.pth.
- Runs inference on a random sample of validation data.
- Prints a clean table comparing Actual Price vs. Predicted Price.

## Troubleshooting & Tips

We faced several hurdles getting the GPU environment stable. Here are some tips to avoid them:

### 1. Torch not compiled with CUDA enabled:

- This happens if you just run pip install torch. You must install the specific CUDA version.  
  Use the command given in requirements.txt or run:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Image Loading Errors:

- Ensure your data/images folder is structured simply as a list of files (1234.jpg, 1235.jpg).
- If the code crashes on a missing image, dataset.py is designed to skip it or handle it gracefully—check your terminal logs for warnings.

## Dataset Description

The dataset used in this project is provided as an Excel file and contains structured tabular data for residential property valuation.

- **File:** `data/train(1).xlsx`, `data/test2.xlsx`
- **Rows:** ~16,200 + 5,400
- **Columns:** 21
- **Format:** Excel (`.xlsx`)
- **Task:** Supervised learning (Regression – Property Price Prediction)

---

## Column Description

| Column Name | Description |
|------------|------------|
| `id` | Unique identifier for each property |
| `date` | Date when the property was sold |
| `price` | **Target variable** — sale price of the property |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Living area size (in square feet) |
| `sqft_lot` | Total land area (in square feet) |
| `floors` | Number of floors in the property |
| `waterfront` | Indicates whether the property has a waterfront view (1 = Yes, 0 = No) |
| `view` | Quality of the view from the property |
| `condition` | Overall condition rating of the property |
| `grade` | Overall construction and design grade |
| `sqft_above` | Square footage of the property above ground |
| `sqft_basement` | Square footage of the basement |
| `yr_built` | Year the property was originally built |
| `yr_renovated` | Year of the most recent renovation (0 if never renovated) |
| `zipcode` | ZIP code of the property location |
| `lat` | Latitude coordinate |
| `long` | Longitude coordinate |
| `sqft_living15` | Average living area size of nearby properties |
| `sqft_lot15` | Average lot size of nearby properties |

---

## Data Notes

- The dataset is included directly in the repository for reproducibility
- Missing or zero values (e.g., `yr_renovated = 0`) are handled during preprocessing
- Numerical features are scaled before model training
- Geographic features (`lat`, `long`) are used for spatial understanding

---

## Reproducibility

All experiments and model training steps in this repository can be reproduced using the provided dataset and scripts.

_*Further details are provided inside .py files itself.  
  Model is trained without lat n long, and they are used only for the data\images\ 
  one can modify or add the followin columns to make the model work more acc.*_
