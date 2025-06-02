# MultiClass Classification Of Diabetic Retinopathy using Swin V2.
This Repo contains the Diabetic Retinopathy Classification Done by Modified Swin V2 with Selective Reattention Branch with higher performance than baseline model.

Here we used Different Image Processing Techniques like CLAHE, Adaptive Gamma Correction and normalization etc. 
The Main Work in Here is Modified Swinv2 in which the #Window Attention Module is inherited and overrided.
We Addded an uncertainty branch which Calculating the window Attention, and then it undergoes the MLP and give a Uncertainty Score it is then used in the attention to rescale the Window Attention.

This Model acheives 97% Accuracy on Testing Data from APTOS 2019 which is 10% (approx) higher than the baseline model with same hyper parameters.
Modules
📦 Data Augmentation Module
Applies standard transformations like flipping, rotation, brightness, contrast, etc.

Enhances dataset diversity and addresses class imbalance.

🖼 Image Processing Module
Performs CLAHE (Contrast Limited Adaptive Histogram Equalization).

Uses adaptive gamma correction to enhance visibility of retinal features.

Resizes all images to 256×256 and normalizes them for training.

🧩 Swin Backbone Module
Swin-Tiny Transformer is used as the base architecture.

Window-based attention enables both local and global feature learning.

Backbone is frozen during UGA training for better generalization.

❓ Uncertainty Module
Adds a parallel branch to estimate prediction uncertainty.

Helps the model attend better to ambiguous regions.

Outputs a scalar uncertainty score for each prediction (with potential for patch-wise extension).

📊 Evaluation Module
Evaluates performance using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

Uses Grad-CAM for visualizing attention maps and identifying model focus regions.

✅ Highlights
Fine-grained 5-class DR classification: No_DR, Mild, Moderate, Severe, Proliferative_DR.

Significantly improved performance over the baseline Swin model.

Designed specifically for the APTOS 2019 dataset.

Better interpretability using Grad-CAM and Uncertainty visualization.

📈 Results
Model	Accuracy	Key Benefit
Swin-Tiny (Baseline)	88.83%	Good baseline for DR classification
UGA-Swin (Ours)	97.34%	Excellent handling of confusing stages

⚠️ Known Limitations
The architecture is optimized for the APTOS 2019 dataset and may require fine-tuning for other datasets.

The newly introduced uncertainty layers are not pretrained due to lack of ImageNet-compatible features, increasing training resource requirements.

Grad-CAM visualizations sometimes pick up features from irrelevant black border regions.

The uncertainty module currently provides a single scalar value; future versions could expand to patch-wise uncertainty maps (e.g., 8×8 grid).

🔮 Future Work
Implement patch-wise uncertainty maps for better attention granularity.

Train and validate the model on external datasets like Messidor and EyePACS for generalization.

Integrate into a clinical decision support system with real-time inference and doctor-friendly visualizations.

