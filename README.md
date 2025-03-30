1. Introduction
This project, "Building a Hand Sign Language Recognition System," aims to leverage deep learning techniques to bridge communication gaps for the deaf and hard-of-hearing community. According to the World Health Organization (WHO), over 466 million people worldwide suffer from hearing loss, with 34 million being children under 15. Sign language, a visual means of communication using hand gestures, facial expressions, and body movements, is vital for this community. However, the lack of widespread understanding of sign language creates barriers in daily interactions.

Our solution focuses on Isolated Sign Language (ISL) recognition, specifically targeting the American Sign Language (ASL) alphabet, using real-time camera input. By deploying deep learning models, we aim to translate hand signs into readable text, facilitating seamless communication between deaf individuals and non-signers.

2. Dataset
We used the ASL Alphabet Dataset, which consists of:

Training set: 43,500 images (200×200 pixels) across 29 classes (A-Z, SPACE, DELETE, NOTHING).

Test set: 29 images (one per class).

Data distribution:

Training (80%): 34,800 images

Validation (10%): 4,350 images

Test (10%): 4,350 images

Preprocessing & Augmentation:
To enhance model generalization and prevent overfitting, we applied:

ImageDataGenerator (TensorFlow) for transformations:

Random rotation, zoom, brightness adjustment, horizontal flips.

Normalization (pixel values scaled to [0, 1]).

Stratified sampling to maintain class balance.

3. Models & Methodology
We experimented with three state-of-the-art CNN architectures using transfer learning:

a) ResNet50
Architecture: 50-layer residual network pre-trained on ImageNet.

Modifications:

Frozen base layers, added GlobalAveragePooling2D + Dense (1024 ReLU) + Softmax (29 classes).

Training:

Batch size: 64

Epochs: 5

Optimizer: Adam

Results:

Test accuracy: 27% (low performance, likely due to insufficient fine-tuning).

b) EfficientNetB0
Architecture: Lightweight, compound-scaled CNN optimized for efficiency.

Modifications:

Added Dropout (0.5) + Dense (512 ReLU) + Softmax.

Training: Same as ResNet50.

Results:

Test accuracy: 3% (poor fit, potentially due to limited data for fine-tuning).

c) InceptionV3
Architecture: 42-layer CNN with parallel convolution paths (1×1, 3×3, 5×5 filters).

Modifications:

Frozen base, added GlobalAveragePooling2D + Dense (1024 ReLU) + Softmax.

Fine-tuning: Unlocked top Inception blocks, trained with SGD (lr=0.0001, momentum=0.9).

Training:

Batch size: 64

Epochs: 5 (initial training) + 5 (fine-tuning).

Results:

Test accuracy: 98% (highest performance).

Confusion matrix: Showed minimal misclassifications.

Real-time demo: Successfully predicted signs from webcam feed (90.49% confidence for "E").

4. Tools & Libraries
Languages: Python

Frameworks: TensorFlow, Keras, OpenCV

Data Augmentation: ImageDataGenerator

Model Deployment: Real-time prediction via webcam.

5. Key Findings & Challenges
InceptionV3 outperformed ResNet50 and EfficientNet due to its multi-scale feature extraction (parallel convolutions).

Challenges:

Low accuracy in ResNet/EfficientNet due to shallow fine-tuning.

Limited dataset diversity (e.g., variations in hand lighting/angles).

Future Work:

Integrate RNN/LSTM for Continuous Sign Language (CSL) recognition.

Expand dataset to include more signers and environments.

Optimize for edge devices (e.g., mobile deployment).

6. Conclusion
This project demonstrates the potential of deep learning for sign language recognition, achieving 98% accuracy with InceptionV3. By improving accessibility, this system can empower the deaf community and foster inclusive communication.  Future enhancements will focus on real-time sentence translation and broader language support.
