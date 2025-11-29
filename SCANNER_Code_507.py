# ============================================================
# SCANNER: A Deep Learning Method for Multi-Class Skin Lesion Classification Using Explainable AI 
# Author: Om Patel
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
import seaborn as sns
import cv2
import warnings
warnings.filterwarnings('ignore')

# Create directory for saving outputs
output_dir = 'model_outputs'
os.makedirs(output_dir, exist_ok=True)

# Paths of the skin lesion images
train_dir = r"HAM10000_images_part_1"
metadata_path = r"HAM10000_metadata - HAM10000_metadata.csv"

# Check if required files and directories exist
print("Checking file paths:----------")
if not os.path.exists(train_dir):
    print(f"Warning: Training directory not found: {train_dir}")
else:
    print(f"Training directory found: {train_dir}")

if not os.path.exists(metadata_path):
    print(f"Error: Metadata file not found: {metadata_path}")
    exit()
else:
    print(f"Metadata file found: {metadata_path}")

# Load and examine the metadata file
try:
    meta = pd.read_csv(metadata_path)
    print("\nMetadata loaded successfully")
    print(f"Total records in metadata: {len(meta)}")
    print(meta.head())
except FileNotFoundError:
    print(f"Error: Metadata file not found at '{metadata_path}'")
    print("Please check the file path and try again.")
    exit()
except Exception as e:
    print(f"Error loading metadata: {e}")
    exit()

# A function to locate image files in train directory
def get_image_path(image_id):
    train_path = os.path.join(train_dir, image_id + ".jpg")
    if os.path.exists(train_path):
        return train_path
    
    return None


print("\nFinding image files:----------")
meta["image_path"] = meta["image_id"].apply(get_image_path)

# A report on how many images were found
total_records = len(meta)
found_images = meta["image_path"].notna().sum()
missing_images = total_records - found_images

print(f"Total metadata records: {total_records}")
print(f"Images found: {found_images}")
print(f"Images missing: {missing_images}")

# Remove records without valid image paths
meta = meta[meta["image_path"].notna()]

if len(meta) == 0:
    print("\nError: No images found. Please check your directory paths.")
    print(f"Expected directories:")
    print(f"  - Training: {train_dir}")
    exit()

print(f"\nWorking with {len(meta)} images that have valid paths")

# Convert diagnosis labels to numerical format and create medical name mapping
label_encoder = LabelEncoder()
meta["label_enc"] = label_encoder.fit_transform(meta["dx"])
class_labels = list(label_encoder.classes_)
num_classes = len(class_labels)

# Medical name mapping for better readability
medical_names = {
    'akiec': 'Actinic Keratosis (AKIEC)',
    'bcc': 'Basal Cell Carcinoma (BCC)',
    'bkl': 'Benign Keratosis (BKL)',
    'df': 'Dermatofibroma (DF)',
    'mel': 'Melanoma (MEL)',
    'nv': 'Melanocytic Nevus (NV)',
    'vasc': 'Vascular Lesion (VASC)'
}

print(f"\nDiagnosis classes found: {class_labels}")
print("Medical condition names:")
for code, name in medical_names.items():
    print(f"  {code} -> {name}")

# Split data into training and testing sets
print("\nSplitting dataset into training and test sets:----------")
train_df, test_df = train_test_split(meta, test_size=0.2, stratify=meta["label_enc"], random_state=42)
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# A function that balances our skin cancer dataset so that rare types get equal attention
def create_enhanced_balanced_generator(df, target_col='dx'):
    print("Creating enhanced balanced training dataset:----------")
    
    # Calculate class weights to handle the imbalance
    class_counts = df[target_col].value_counts()
    total_samples = len(df)
    class_weights = {}
    
    for class_name in class_labels:
        count = class_counts.get(class_name, 0)
        if count > 0:
            class_weights[class_name] = (total_samples / (len(class_labels) * count)) ** 0.5
        else:
            class_weights[class_name] = 1.0
    
    print("Enhanced class weights for training:")
    for class_name, weight in class_weights.items():
        print(f"  {medical_names[class_name]:<35}: {weight:.2f}x")
    
    balanced_dfs = []
    
    for class_name in df[target_col].unique():
        class_df = df[df[target_col] == class_name]
        current_count = len(class_df)

        if class_name == 'nv':  
            target_samples = min(800, current_count) 
        elif class_name in ['bkl', 'mel']:  
            target_samples = min(700, current_count)
        elif class_name in ['bcc', 'akiec']:  
            target_samples = 600
        else:  
            target_samples = 500
        
        if current_count < target_samples:
            oversampled = class_df.sample(target_samples, replace=True, random_state=42)
        else:
            oversampled = class_df.sample(target_samples, random_state=42)
            
        balanced_dfs.append(oversampled)
        print(f"  {medical_names[class_name]:<35}: {current_count:>3} -> {len(oversampled):>3} samples")
    
    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42)
    
    print(f"Final enhanced balanced dataset size: {len(balanced_df)} samples")
    return balanced_df, class_weights

# Create the enhanced balanced training dataset
balanced_train_df, class_weights = create_enhanced_balanced_generator(train_df)

# Image processing configuration
IMG_SIZE = 224  
BATCH_SIZE = 32

# Opting data augmentation for training images
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    channel_shift_range=0.1,
    fill_mode='reflect',
    validation_split=0.15
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Training the data generator
train_gen = train_datagen.flow_from_dataframe(
    balanced_train_df,
    x_col="image_path",
    y_col="dx",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset='training'
)

# Validating the data generator
val_gen = train_datagen.flow_from_dataframe(
    balanced_train_df,
    x_col="image_path",
    y_col="dx",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=False
)

# Testing the data generator
test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col="image_path",
    y_col="dx",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Build the neural network model using ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Keep pre-trained weights fixed

# Add specialized layers for skin cancer detection
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.4),
    BatchNormalization(),
    Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

# Learning rules with careful pace
initial_learning_rate = 0.001
model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# Prevents overlearning by stopping when progress plateaus
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    restore_best_weights=True,
    min_delta=0.001,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=6,
    min_lr=1e-7,
    verbose=1
)

# Automatically keeps the best-performing version as training progresses
checkpoint = ModelCheckpoint(
    os.path.join(output_dir, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Display the model architecture
model.summary()

# Begins the training process
print("\nStarting  training: ------")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
    class_weight=class_weights
)

# Restores the model at its most accurate moment.
model.load_weights(os.path.join(output_dir, 'best_model.h5'))
print("Loaded best model weights from checkpoint")

# PERFORMANCE HEALTH CHECK
print("PERFORMANCE HEALTH CHECK")

# Seeing how well our model diagnoses
test_gen.reset()
test_preds = model.predict(test_gen, verbose=1)
test_pred_classes = np.argmax(test_preds, axis=1)
test_true_classes = test_gen.classes
test_true_labels = tf.keras.utils.to_categorical(test_true_classes, num_classes=num_classes)

# Calculate multiple metrics
precision, recall, f1, support = precision_recall_fscore_support(
    test_true_classes, test_pred_classes, average=None
)

# Macro and weighted averages
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    test_true_classes, test_pred_classes, average='macro'
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    test_true_classes, test_pred_classes, average='weighted'
)

# Additional metrics
balanced_acc = balanced_accuracy_score(test_true_classes, test_pred_classes)
kappa = cohen_kappa_score(test_true_classes, test_pred_classes)
mcc = matthews_corrcoef(test_true_classes, test_pred_classes)

# Check the performance on validation set
print("\nValidation Set Performance:----------")
val_loss, val_acc, val_precision, val_recall = model.evaluate(val_gen)
print(f"Validation Accuracy: {val_acc:.4f}")

# Check the performance on test set
print("\nTest Set Performance:----------")
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.4f}")

# Enhanced comprehensive performance display
print(f"{'COMPREHENSIVE PERFORMANCE METRICS BY CLASS':^80}")
print(f"{'Medical Diagnosis':<35} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")

for i, class_name in enumerate(class_labels):
    precision_val = precision[i]
    recall_val = recall[i]
    f1_val = f1[i]
    support_val = support[i]
    medical_name = medical_names[class_name]
    print(f"{medical_name:<35} {precision_val:.4f}    {recall_val:.4f}    {f1_val:.4f}    {support_val:<10}")

print(f"{'MACRO AVERAGE':<35} {precision_macro:.4f}    {recall_macro:.4f}    {f1_macro:.4f}    {len(test_true_classes):<10}")
print(f"{'WEIGHTED AVERAGE':<35} {precision_weighted:.4f}    {recall_weighted:.4f}    {f1_weighted:.4f}    {len(test_true_classes):<10}")
print()
print()
print()

# Detailed performance breakdown
print(f"\n{'ADDITIONAL OVERALL METRICS':^80}")
print(f"{'Balanced Accuracy':<25}: {balanced_acc:.4f}")
print(f"{'Cohen Kappa Score':<25}: {kappa:.4f}")
print(f"{'Matthews Correlation':<25}: {mcc:.4f}")
print(f"{'Overall Test Accuracy':<25}: {test_acc:.4f}")
print(f"{'Overall Test Precision':<25}: {test_precision:.4f}")
print(f"{'Overall Test Recall':<25}: {test_recall:.4f}")

# Grading our model's diagnostic ability
print(f"\n{'PERFORMANCE INTERPRETATION':^80}")
if test_acc >= 0.85:
    performance_level = "EXCELLENT"
elif test_acc >= 0.75:
    performance_level = "GOOD"
elif test_acc >= 0.65:
    performance_level = "MODERATE"
else:
    performance_level = "NEEDS IMPROVEMENT"

if kappa >= 0.8:
    agreement_level = "ALMOST PERFECT"
elif kappa >= 0.6:
    agreement_level = "SUBSTANTIAL"
elif kappa >= 0.4:
    agreement_level = "MODERATE"
else:
    agreement_level = "FAIR"

print(f"{'Accuracy Level':<25}: {performance_level}")
print(f"{'Agreement Level (Kappa)':<25}: {agreement_level}")
print(f"{'Model Reliability':<25}: {'HIGH' if balanced_acc > 0.75 and f1_macro > 0.75 else 'MODERATE'}")

# SEPARATE VISUALIZATIONS
# 1. Confusion Matrix
print("\nGenerating Confusion Matrix Visualization:----------")
cm = confusion_matrix(test_true_classes, test_pred_classes)
medical_labels = [medical_names[cls] for cls in class_labels]

plt.figure(figsize=(14, 12))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

sns.heatmap(cm_normalized, annot=True, fmt=".1%", cmap="Blues", 
            xticklabels=medical_labels, yticklabels=medical_labels,
            cbar_kws={'label': 'Recall Percentage'})
plt.title(f'Confusion Matrix - Test Set Performance\nOverall Accuracy: {test_acc:.2%}', 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Predicted Diagnosis', fontsize=14, fontweight='bold')
plt.ylabel('Actual Diagnosis', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Performance metrics visualization
print("\nGenerating Performance Metrics Visualization:----------")
plt.figure(figsize=(16, 10))

metrics_data = {
    'Precision': precision,
    'Recall': recall, 
    'F1-Score': f1
}

x = np.arange(len(class_labels))
width = 0.25
colors = ['#2E86AB', '#A23B72', '#F18F01']

for i, (metric_name, values) in enumerate(metrics_data.items()):
    plt.bar(x + i*width, values, width, label=metric_name, color=colors[i], alpha=0.8)

plt.xlabel('Diagnosis Classes', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.title('Performance Metrics by Skin Lesion Class', fontsize=16, pad=20, fontweight='bold')
plt.xticks(x + width, [medical_names[cls][:20] + '...' for cls in class_labels], 
           rotation=45, ha='right', fontsize=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, 1.1)

# Add value labels on top of the bars
for i, (metric_name, values) in enumerate(metrics_data.items()):
    for j, value in enumerate(values):
        plt.text(j + i*width, value + 0.02, f'{value:.2f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_metrics_by_class.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Training History vvisualization
print("\nGenerating Training History Visualization:----------")
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
plt.title('Model Accuracy During Training', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
plt.title('Model Loss During Training', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Class distribution visualization
print("\nGenerating Class Distribution Visualization:----------")
plt.figure(figsize=(14, 8))
class_counts_train = train_df['dx'].value_counts()
class_counts_test = test_df['dx'].value_counts()
class_counts_balanced = balanced_train_df['dx'].value_counts()

x = np.arange(len(class_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
train_bars = ax.bar(x - width, [class_counts_train.get(cls, 0) for cls in class_labels], width, 
                   label='Original Training', alpha=0.7, color='blue')
balanced_bars = ax.bar(x, [class_counts_balanced.get(cls, 0) for cls in class_labels], width, 
                      label='Balanced Training', alpha=0.7, color='green')
test_bars = ax.bar(x + width, [class_counts_test.get(cls, 0) for cls in class_labels], width, 
                  label='Test Set', alpha=0.7, color='red')

ax.set_xlabel('Diagnosis Classes', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
ax.set_title('Class Distribution Across Datasets', fontsize=16, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([medical_names[cls] for cls in class_labels], rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add value labels on top of the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_value_labels(train_bars)
add_value_labels(balanced_bars)
add_value_labels(test_bars)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. Overall metrics summary
print("\nGenerating Overall Metrics Summary:---------")
plt.figure(figsize=(12, 8))

overall_metrics = [test_acc, val_acc, balanced_acc, f1_macro, kappa]
metric_names = ['Test\nAccuracy', 'Validation\nAccuracy', 'Balanced\nAccuracy', 'Macro\nF1-Score', "Cohen's\nKappa"]
colors_summary = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E885B']

bars = plt.bar(metric_names, overall_metrics, color=colors_summary, alpha=0.8, edgecolor='black')
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.title('Overall Model Performance Metrics', fontsize=16, pad=20, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, overall_metrics):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overall_metrics_summary.png'), dpi=300, bbox_inches='tight')
plt.show()

# SAMPLE PREDICTIONS AND EXPLAINABLE AI COMPONENTS
# A function that automatically outlines skin lesions to visualize what areas the AI is analyzing
def add_lesion_boundary(img, true_label, pred_label):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    
    # Create color masks to identify lesion areas
    lower_dark = np.array([0, 50, 50])
    upper_dark = np.array([15, 255, 255])
    mask1 = cv2.inRange(hsv, lower_dark, upper_dark)
    
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find and draw contours around the lesion
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        min_area = 500
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if significant_contours:
            largest_contour = max(significant_contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            smooth_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            cv2.drawContours(img_rgb, [smooth_contour], -1, (0, 255, 0), 3)
            cv2.drawContours(img_rgb, [smooth_contour], -1, (255, 255, 255), 1)
    
    return cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

# ENHANCED function that shows AI's diagnoses vs actual conditions with highlighted lesions
def show_sample_predictions(generator, num_images=3, title="AI Skin Lesion Diagnosis Results"):
    generator.reset()
    images, labels = next(generator)
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)

    # Create a more visually appealing layout
    fig = plt.figure(figsize=(18, 8))
    
    # Add a main title with better styling
    plt.suptitle(title, fontsize=22, fontweight='bold', y=0.98, color='#2E86AB')
    
    for i in range(min(num_images, len(images))):
        plt.subplot(1, 3, i+1)
        img = (images[i] + 1) / 2
        img = (img * 255).astype(np.uint8)
        
        img_with_boundary = add_lesion_boundary(img.copy(), class_labels[true_classes[i]], class_labels[pred_classes[i]])
        
        plt.imshow(img_with_boundary)
        true_medical_name = medical_names[class_labels[true_classes[i]]]
        pred_medical_name = medical_names[class_labels[pred_classes[i]]]
        confidence = np.max(preds[i])
        is_correct = class_labels[true_classes[i]] == class_labels[pred_classes[i]]
        color = "#2E8B57" if is_correct else "#DC143C"  # Forest Green vs Crimson Red
        
        # Enhanced title with better formatting and symbols
        status = "CORRECT" if is_correct else "INCORRECT"
        confidence_color = "#2E8B57" if confidence > 0.8 else "#FF8C00" if confidence > 0.6 else "#DC143C"
        
        title_text = f"Actual: {true_medical_name}\nPredicted: {pred_medical_name}\nConfidence: {confidence:.1%}\n{status}"
        
        plt.title(title_text, color=color, fontsize=13, pad=20, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F8FF', edgecolor=color, linewidth=3))
        plt.axis("off")
        
        # Add confidence indicator
        plt.text(0.5, -0.15, f"Confidence Level", transform=plt.gca().transAxes, 
                ha='center', fontsize=10, fontweight='bold', color='#333333')
        plt.text(0.5, -0.25, "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low", 
                transform=plt.gca().transAxes, ha='center', fontsize=11, fontweight='bold', 
                color=confidence_color)
    
    # Add a comprehensive legend/explanation at the bottom
    plt.figtext(0.5, 0.02, 
                "Green Contour = AI-Detected Lesion Boundary | "
                "Green Border = Correct Diagnosis | "
                "Red Border = Incorrect Diagnosis\n"
                "Confidence: High (>80%) | Medium (60-80%) | Low (<60%)", 
                ha="center", fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=1.0", facecolor="#E6F3FF", edgecolor="#2E86AB", alpha=0.9))
    
    plt.tight_layout(pad=4.0)
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(os.path.join(output_dir, 'sample_predictions_with_lesions.png'), dpi=300, bbox_inches='tight', 
                facecolor='#F5F5F5', edgecolor='none')
    plt.show()

print("\nGenerating sample predictions with lesion detection:----------")
try:
    test_gen.reset()
    show_sample_predictions(test_gen, num_images=3, title="AI Skin Lesion Diagnosis - Clinical Analysis")
except Exception as e:
    print(f"Error showing sample predictions: {e}")

# A function that creates visual explanations showing exactly which image regions influenced the diagnosis
def generate_saliency_map(model, img_path, sample_num):
    try:
        print(f"\nAnalyzing: {os.path.basename(img_path)}")
        
        # Load and prepare image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_array_processed = preprocess_input(img_array_expanded.copy())
        
        # Get true diagnosis
        image_id = os.path.basename(img_path).replace('.jpg', '')
        true_label_row = meta[meta['image_id'] == image_id]
        true_label = true_label_row['dx'].values[0] if len(true_label_row) > 0 else "Unknown"
        
        # Make prediction
        preds = model.predict(img_array_processed, verbose=0)
        pred_class_idx = np.argmax(preds[0])
        pred_class = class_labels[pred_class_idx]
        confidence = preds[0][pred_class_idx]
        
        print(f"Diagnosis Results:")
        print(f"=> Actual: {medical_names.get(true_label, true_label)}")
        print(f"=> Predicted: {medical_names[pred_class]}")
        print(f"=> Confidence: {confidence:.2%}")
        
        # Compute which image regions influenced the decision
        img_tensor = tf.cast(img_array_processed, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            top_class_output = predictions[:, pred_class_idx]
        
        # Calculate importance of each pixel
        gradients = tape.gradient(top_class_output, img_tensor)
        gradients = tf.reduce_max(tf.abs(gradients), axis=-1)
        
        # Create saliency map
        saliency = np.abs(gradients[0].numpy())
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Create visualization
        fig = plt.figure(figsize=(18, 12))
        
        gs = plt.GridSpec(2, 3, figure=fig, height_ratios=[1, 0.3], hspace=0.4, wspace=0.3)
        
        # Original image with lesion boundary
        ax1 = fig.add_subplot(gs[0, 0])
        img_with_boundary = add_lesion_boundary(img_array.astype(np.uint8), true_label, pred_class)
        ax1.imshow(img_with_boundary)
        ax1.set_title('Original Image\n(Detected Lesion Boundary)', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.axis('off')
        
        # Saliency heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.imshow(saliency, cmap='hot')
        ax2.set_title('Saliency Heatmap\n(Bright = High Importance)', 
                     fontsize=12, fontweight='bold', pad=10)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Overlay of important regions
        ax3 = fig.add_subplot(gs[0, 2])
        thresholded_saliency = np.where(saliency > 0.7, saliency, 0)
        ax3.imshow(img_array.astype(np.uint8))
        ax3.imshow(thresholded_saliency, cmap='hot', alpha=0.8)
        ax3.set_title('Key Decision Regions\n(Top 30% Important Areas)', 
                     fontsize=12, fontweight='bold', pad=10)
        ax3.axis('off')
        
        # Overall title
        result_color = "green" if true_label == pred_class else "red"
        plt.suptitle(f'AI Decision Explanation: {medical_names[pred_class]}\n'
                    f'Confidence: {confidence:.2%} | Match: {"CORRECT" if true_label == pred_class else "INCORRECT"}', 
                    fontsize=14, fontweight='bold', y=0.95, color=result_color)
        
        # Explanation text
        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis('off')
        
        explanation_text = (
            "WHAT YOU'RE SEEING:\n"
            "=> Saliency maps show which parts of the image the AI focused on for diagnosis\n"
            "=> Bright red/yellow areas = High influence on the decision\n"
            "=> Dark blue areas = Low influence on the decision\n"
            "=> The AI analyzes: lesion shape, color patterns, texture, and border characteristics"
        )
        
        ax_text.text(0.02, 0.6, explanation_text, fontsize=11, 
                    bbox=dict(boxstyle="round,pad=1.0", facecolor="lightblue", alpha=0.7),
                    verticalalignment='center', horizontalalignment='left',
                    transform=ax_text.transAxes)
        
        # Analysis summary
        summary_text = (
            f"ANALYSIS SUMMARY:\n"
            f"Maximum saliency: {saliency.max():.3f}\n"
            f"Average saliency: {saliency.mean():.3f}\n"
            f"Focus area ratio: {(saliency > 0.5).sum() / saliency.size:.1%}"
        )
        
        ax_text.text(0.65, 0.6, summary_text, fontsize=10,
                    bbox=dict(boxstyle="round,pad=1.0", facecolor="lightgreen", alpha=0.7),
                    verticalalignment='center', horizontalalignment='left',
                    transform=ax_text.transAxes)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.4, wspace=0.3)
        plt.savefig(os.path.join(output_dir, f'saliency_map_sample_{sample_num}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Console summary
        print(f"\nSaliency Analysis Summary:----------")
        print(f"=> Maximum saliency: {saliency.max():.3f}")
        print(f"=>nAverage saliency: {saliency.mean():.3f}")
        print(f"=> Focus area ratio: {(saliency > 0.5).sum() / saliency.size:.1%}")
        
        if true_label == pred_class:
            print(f"   AI correctly identified the lesion type")
        else:
            print(f"   AI misdiagnosed - review recommended")
        
        return True
        
    except Exception as e:
        print(f"Error in saliency analysis: {e}")
        return False

# A function that systematically analyzes multiple cases to reveal the visual reasoning behind each diagnosis
def run_saliency_analysis(model, test_df, num_samples=2):
    print("SALIENCY MAP ANALYSIS - AI DECISION EXPLANATION")
    print()
    print("Purpose: Understand which image regions influenced the AI's diagnosis")
    print("How to read: Brighter colors = More important for the decision")
    print("Goal: Verify the AI is focusing on clinically relevant lesion features")
    print("="*70)
    
    samples = test_df.sample(min(num_samples, len(test_df)))
    
    successful = 0
    for idx, (_, row) in enumerate(samples.iterrows()):
        print(f"SAMPLE {idx + 1} of {len(samples)}")
        
        if generate_saliency_map(model, row['image_path'], idx + 1):
            successful += 1
        
        if idx < len(samples) - 1:
            print(f"\nMoving to next sample:----------")

try:
    run_saliency_analysis(model, test_df, num_samples=2)
except Exception as e:
    print(f"Saliency analysis failed: {e}")

# Final comprehensive results summary
print(f"\n{'FINAL COMPREHENSIVE RESULTS SUMMARY':^80}")
print(f"{'Metric':<25} {'Score':<15} {'Interpretation':<30}")
print(f"{'Test Accuracy':<25} {test_acc:.4f}          {'EXCELLENT' if test_acc > 0.85 else 'GOOD' if test_acc > 0.75 else 'MODERATE'}")
print(f"{'Validation Accuracy':<25} {val_acc:.4f}          {'EXCELLENT' if val_acc > 0.85 else 'GOOD' if val_acc > 0.75 else 'MODERATE'}")
print(f"{'Balanced Accuracy':<25} {balanced_acc:.4f}          {'BALANCED' if balanced_acc > 0.75 else 'IMBALANCED'}")
print(f"{'Macro F1-Score':<25} {f1_macro:.4f}          {'STRONG' if f1_macro > 0.75 else 'MODERATE'}")
print(f"{'Cohen Kappa':<25} {kappa:.4f}          {'SUBSTANTIAL' if kappa > 0.6 else 'MODERATE'}")
print(f"{'Clinical Reliability':<25} {'HIGH' if test_acc > 0.85 and f1_macro > 0.80 else 'MODERATE'}")


# Save the final trained model
model.save(os.path.join(output_dir, 'enhanced_skin_lesion_classifier.h5'))
print(f"\nModel saved as '{output_dir}/enhanced_skin_lesion_classifier.h5'")
print(f"\nAll analysis graphs and outputs have been saved to the '{output_dir}' directory:")
print(f"\nFINAL MODEL PERFORMANCE: {test_acc:.2%} Test Accuracy | {val_acc:.2%} Validation Accuracy")