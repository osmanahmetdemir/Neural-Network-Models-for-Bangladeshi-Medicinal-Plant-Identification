# Utilizing Neural Network Models to Leaf Images to Identify Bangladeshi Medicinal Plants

## Abstract
In this project, we evaluated convolutional neural network (CNN) architectures using a dataset of leaf images for the identification of medicinal plants in Bangladesh. The datasets consist of 10 classes, each corresponding to a specific plant species: **Phyllanthus emblica**, **Terminalia arjuna**, **Kalanchoe pinnata**, **Centella asiatica**, **Justicia adhatoda**, **Mikania micrantha**, **Azadirachta indica**, **Hibiscus rosa-sinensis**, **Ocimum tenuiflorum**, and **Calotropis gigantea**. The dataset contains a total of 2,029 original leaf images, accompanied by an additional 38,606 augmented images.

Initially, our goal was to achieve acceptable accuracy values by applying a custom CNN architecture to both the normal dataset and its augmented counterpart. Additionally, we aimed to achieve acceptable accuracy values by applying at least two different pre-trained CNN models, such as VGG, Inception, and DenseNet, to both datasets. We implemented these models and compared their accuracy values, generating confusion matrices, training/validation loss, and accuracy values, as well as tables showing precision, recall, and F1 results for each class.

## Materials

### BDMediLeaves Dataset
The dataset named **BDMediLeaves** consists of a collection of images obtained from numerous nurseries and botanical gardens located in Dhaka, Bangladesh. The original dataset consists of approximately 2,000 images, and the augmented dataset contains around 38,000 images. The dataset is divided into three subdirectories: train, test, and validation. Each image in the augmented dataset has a size of approximately 512 × 512 pixels.

The dataset includes data on 10 different leaf types:
- **Phyllanthus emblica**
- **Terminalia arjuna**
- **Kalanchoe pinnata**
- **Centella asiatica**
- **Justicia adhatoda**
- **Mikania micrantha**
- **Azadirachta indica**
- **Hibiscus rosa-sinensis**
- **Ocimum tenuiflorum**
- **Calotropis gigantea**

The images are color images in RGB format.

## Results

### Original Dataset with CNNs, VGG, and Inception Architectures
The BDMediLeaves dataset contains nearly 2,100 leaf images across 10 classes. The dataset was randomly divided into 3 main groups: 70% training, 20% validation, and 10% testing. We loaded the dataset and customized it using different ImageDataGenerators for each model. Each model was trained with specific batch size, target size, and epoch values, with training halting at an appropriate epoch. We tested the models on the test dataset, calculated performance metrics, and plotted graphs for training and validation accuracy and loss. Test accuracy was printed, and the confusion matrix was generated.

#### A- CNN
#### B- Inception
#### C- VGG

The most suitable batch, epoch, and target sizes for each model were determined, and the results are tabulated for comparison:

| Model      | Precision | Recall | F1 Score | Accuracy | Batch Size | Epoch | Target Size |
|------------|-----------|--------|----------|----------|------------|-------|-------------|
| CNN        | 0.69      | 0.67   | 0.67     | 0.68     | 32         | 30    | 128x128     |
| Inception  | 0.96      | 0.96   | 0.95     | 0.96     | 32         | 10    | 224x224     |
| VGG-19     | 0.84      | 0.83   | 0.82     | 0.83     | 32         | 30    | 256x256     |

According to the table, the best model was determined to be Inception, followed by VGG-19, with CNN yielding the least favorable results.

### Augmented Dataset with CNNs, DenseNet, and Inception Architectures
The augmented BDMediLeaves dataset contains 38,606 leaf images across 10 classes. The dataset was divided into 3 main groups: 75% training, 15% validation, and 10% testing. The training process was repeated for each model according to different batch sizes and epochs.

#### A- CNN
#### B- Inception
#### C- DenseNet

The most suitable batch, epoch, and target sizes for each model were determined, and the results are tabulated for comparison:

| Model      | Precision | Recall | F1 Score | Accuracy | Batch Size | Epoch | Target Size |
|------------|-----------|--------|----------|----------|------------|-------|-------------|
| CNN        | 0.94      | 0.93   | 0.93     | 0.93     | 32         | 12    | 256x256     |
| Inception  | 0.95      | 0.94   | 0.94     | 0.94     | 32         | 7     | 299x299     |
| DenseNet   | 0.87      | 0.86   | 0.86     | 0.86     | 32         | 8     | 256x256     |

According to the table, the best model was determined to be Inception, followed by CNN, with DenseNet yielding the least favorable results.

## Conclusion
Using 4 different models on both the original dataset of 2,029 images and the augmented dataset of 38,606 images, we experimented with various parameters such as epoch, batch size, and target size to find the best performance. The Inception model showed the best performance on both datasets.

