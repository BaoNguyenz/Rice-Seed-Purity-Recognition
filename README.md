# Enhancing Rice Seed Purity Recognition Accuracy Based on Optimal Feature Selection  

## Introduction  
This project offers a robust and efficient solution for classifying rice seed purity to meet agricultural standards. It employs advanced image processing techniques, optimal feature selection methods, and machine learning algorithms to enhance accuracy and efficiency. This tool is designed to assist agricultural stakeholders in ensuring seed quality, productivity, and economic value.  

## Data description

The dataset used in this study consists of images from six prominent rice varieties commonly grown in northern Vietnam: **BC-15, Huong Thom-1, Nep-87, Q-5, Thien Uu-8, and Xi-23**. To create a balanced and practical model, the dataset is divided into two subsets for each variety:

- **Positive Subset**: Contains images of seeds that belong to the target rice variety (e.g., the BC-15 subset includes only BC-15 seeds).  
- **Negative Subset**: Includes images of seeds from other rice varieties or non-rice seeds to simulate potential misclassification scenarios (e.g., the BC-15 subset includes seeds from Huong Thom-1, Nep-87, etc.).

This structured dataset reflects the real-world challenges faced by rice seed production companies, improving the model's ability to accurately classify rice varieties. The detailed distribution of images across subsets is provided in **Table 1**.

| Rice Seed Name  | Training Set | Testing Set | Total Images |
|-----------------|--------------|-------------|--------------|
| BC-15           | 2573         | 1104        | 3677         |
| Huong Thom-1    | 2905         | 1245        | 4150         |
| Nep-87          | 2011         | 862         | 2873         |
| Q-5             | 2107         | 903         | 3010         |
| Thien Uu-8      | 1404         | 602         | 2006         |
| Xi-23           | 2901         | 1244        | 4145         |

**Data are available in the link below: [Google Drive](https://drive.google.com/drive/folders/1LqkUIPmZRlam6CFkcRR37BznsMHVV2tn)**

## Installation  
Follow these steps to set up and run the project:  
1. **Clone the repository**:  
   ```bash
   https://github.com/BaoNguyenz/2024_ecological_Rice_seed.git
   cd 2024_ecological_Rice_seed

## Usage  
1. **Feature Extraction**:  
   - Navigate to the `Features_extraction` directory.  
   - Install the required dependencies:  
     ```bash
     pip install -r Features_extraction/requirements.txt
     ```  
   - Run the `Extract_features.ipynb` notebook to extract features from your rice seed dataset.  

2. **Feature Selection**:  
   - Navigate to the `Features_selection_method` directory.  
   - Install the required dependencies:  
     ```bash
     pip install -r Features_selection_method/requirements.txt
     ```  
   - Use one of the feature selection methods provided:  
     - `Filter_method.ipynb`  
     - `Wrapper_method.ipynb`  
     - `Embedded_method.ipynb`  

3. **Train Deep Learning Models on Kaggle**:  
   - Upload the `Deep_Learning_model` folder to Kaggle.  
   - Choose a model from the available options:  
     - `VGG16_model`
     - `ResNet50_model` 
     - `InceptionV3_model`
     - `Dense121_model`
     - `EfficientNetB0_model` 
   - Train the selected model using Kaggle's GPU P100 resources.

## Project Structure

```
.
├── 2024_Ecological_RiceSeed
│   ├── Features_extraction
│   │   ├── Extract_features.ipynb
│   │   └── requirements.txt
│   ├── Features_selection_method
│   │   ├── catboost_info
│   │   ├── Embedded_method.ipynb
│   │   ├── Filter_method.ipynb
│   │   ├── Wrapper_method.ipynb
│   │   └── requirements.txt
│   ├── Deep_Learning_model
│   │   ├── VGG16_model
│   │   │   ├── tranferlearning-bc15-vgg16.ipynb
│   │   │   ├── tranferlearning-huongthom-vgg16.ipynb
│   │   │   ├── tranferlearning-nep87-vgg16.ipynb
│   │   │   ├── tranferlearning-q5-vgg16.ipynb
│   │   │   ├── tranferlearning-thienuu-vgg16.ipynb
│   │   │   └── tranferlearning-xi23-vgg16.ipynb
│   │   ├── ResNet50_model
│   │   │   ├── tranferlearning-bc15-resnet50.ipynb
│   │   │   ├── tranferlearning-huongthom-resnet50.ipynb
│   │   │   ├── tranferlearning-nep87-resnet50.ipynb
│   │   │   ├── tranferlearning-q5-resnet50.ipynb
│   │   │   ├── tranferlearning-thienuu-resnet50.ipynb
│   │   │   └── tranferlearning-xi23-resnet50.ipynb
│   │   ├── InceptionV3_model
│   │   │   ├── tranferlearning-bc15-inceptionv3.ipynb
│   │   │   ├── tranferlearning-huongthom-inceptionv3.ipynb
│   │   │   ├── tranferlearning-nep87-inceptionv3.ipynb
│   │   │   ├── tranferlearning-q5-inceptionv3.ipynb
│   │   │   ├── tranferlearning-thienuu-inceptionv3.ipynb
│   │   │   └── tranferlearning-xi23-inceptionv3.ipynb
│   │   ├── Dense121_model
│   │   │   ├── tranferlearning-bc15-densenet121.ipynb
│   │   │   ├── tranferlearning-huongthom-densenet121.ipynb
│   │   │   ├── tranferlearning-nep87-densenet121.ipynb
│   │   │   ├── tranferlearning-q5-densenet121.ipynb
│   │   │   ├── tranferlearning-thienuu-densenet121.ipynb
│   │   │   └── tranferlearning-xi23-densenet121.ipynb
│   │   ├── EfficientNetB0_model
│   │   │   ├── tranferlearning-bc15-efficientnetb0.ipynb
│   │   │   ├── tranferlearning-huongthom-efficientnetb0.ipynb
│   │   │   ├── tranferlearning-nep87-efficientnetb0.ipynb
│   │   │   ├── tranferlearning-q5-efficientnetb0.ipynb
│   │   │   ├── tranferlearning-thienuu-efficientnetb0.ipynb
│   │   │   └── tranferlearning-xi23-efficientnetb0.ipynb
```
 