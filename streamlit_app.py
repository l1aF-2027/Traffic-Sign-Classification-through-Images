import streamlit as st
import joblib
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from skimage.feature import hog as skimage_hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Constants
PATH_JOBLIB = 'joblib/'
WEIGHTS_OPTIONS = ['Uniform', 'Distance']
METRICS_OPTIONS = ['Chi-Square', 'Correlation', 'Bhattacharyya', 'Intersection', 'Euclidean']
MAPPING = {
    'Nguyhiem': 'Nguy hiểm',
    'Cam': 'Cấm',
    'Chidan': 'Chỉ dẫn',
    'Hieulenh': 'Hiệu lệnh',
    'Phu': 'Phụ'
}

class FeatureExtractor:
    @staticmethod
    def color_histogram(image, bins=8):
        """Extract color histogram features from image"""
        row, column, channel = image.shape[:3]
        size = row * column
        features = []
        for k in range(channel):
            histogram = np.squeeze(cv2.calcHist([image], [k], None, [bins], [0, 256]))
            histogram = histogram / size
            features.extend(histogram)
        return features

    @staticmethod
    def hog(image):
        """Extract HOG features from image"""
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return skimage_hog(
            image, 
            orientations=9, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), 
            visualize=False, 
            block_norm='L2-Hys', 
            transform_sqrt=True
        )

    @classmethod
    def extract_features(cls, images):
        """Combine color histogram and HOG features"""
        color_features = [cls.color_histogram(image) for image in images]
        hog_features = [cls.hog(image) for image in images]
        return [np.concatenate((c, h)) for c, h in zip(color_features, hog_features)]

class DistanceMetrics:
    @staticmethod
    def chi_square(x, y):
        return cv2.compareHist(np.float32(x), np.float32(y), cv2.HISTCMP_CHISQR)

    @staticmethod
    def correlation(x, y):
        return 1 - cv2.compareHist(np.float32(x), np.float32(y), cv2.HISTCMP_CORREL)

    @staticmethod
    def bhattacharyya(x, y):
        return cv2.compareHist(np.float32(x), np.float32(y), cv2.HISTCMP_BHATTACHARYYA)

    @staticmethod
    def intersection(x, y):
        return 1 - cv2.compareHist(np.float32(x), np.float32(y), cv2.HISTCMP_INTERSECT)

    @staticmethod
    def euclidean(x, y):
        return np.linalg.norm(np.float32(x) - np.float32(y))

class Visualizer:
    @staticmethod
    def plot_confusion_matrix(cm, model_name, label_encoder):
        st.markdown(f"<h6>Confusion Matrix - {model_name}</h6>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        plt.colorbar(cax)
        
        classes = list(label_encoder.classes_)
        ax.set_xticklabels([''] + classes, rotation=45)
        ax.set_yticklabels([''] + classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        for (i, j), val in np.ndenumerate(cm):
            text_color = 'white' if val > np.max(cm)/2 else 'black'
            ax.text(j, i, val, ha='center', va='center', color=text_color)
        
        st.pyplot(fig)

    @staticmethod
    def plot_classification_report(y_true, y_pred, labels, model_name):
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).T.round(3).drop(['support'], axis=1)
        
        st.markdown(f"<h6>Classification Report - {model_name}</h6>", unsafe_allow_html=True)
        st.dataframe(df_report)

    @staticmethod
    def plot_pca_variance(pca):
        st.markdown("<h6>PCA Explained Variance Ratio</h6>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        plt.plot(range(1, len(explained_variance_ratio) + 1), 
                cumulative_variance_ratio, 
                'bo-', 
                label='Cumulative Explained Variance Ratio')
        plt.axhline(y=0.95, color='r', linestyle='--', 
                    label='95% Explained Variance Threshold')
        
        plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.legend()
        plt.grid(True)
        
        st.pyplot(fig)

class CustomKernelSVM:
    @staticmethod
    def create_kernel(metric_func):
        def custom_kernel(X, Y=None):
            if Y is None:
                Y = X
            gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
            for i in range(X.shape[0]):
                for j in range(Y.shape[0]):
                    similarity = 1.0 / (1.0 + metric_func(X[i], Y[j]))
                    gram_matrix[i, j] = similarity
            return gram_matrix
        return custom_kernel

def load_models():
    """Load all required models and data"""
    try:
        return {
            'model': joblib.load(f'{PATH_JOBLIB}best_knn_model.joblib'),
            'label_encoder': joblib.load(f'{PATH_JOBLIB}label_encoder.joblib'),
            'train_features': joblib.load(f'{PATH_JOBLIB}train_features.joblib'),
            'test_features': joblib.load(f'{PATH_JOBLIB}test_features.joblib'),
            'train_labels_encoded': joblib.load(f'{PATH_JOBLIB}train_labels_encoded.joblib'),
            'test_labels_encoded': joblib.load(f'{PATH_JOBLIB}test_labels_encoded.joblib')
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

st.set_page_config(
    page_title="Traffic Sign Classification Web",
    page_icon=":vertical_traffic_light:",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>Dự đoán biển báo từ hình ảnh</h1>", 
            unsafe_allow_html=True)

# Load models
model_data = load_models()

# Create metric mappings
metrics_map = {
    'Chi-Square': DistanceMetrics.chi_square,
    'Correlation': DistanceMetrics.correlation,
    'Bhattacharyya': DistanceMetrics.bhattacharyya,
    'Intersection': DistanceMetrics.intersection,
    'Euclidean': DistanceMetrics.euclidean
}

weights_map = {
    'Uniform': 'uniform',
    'Distance': 'distance'
}

# Create layout
col1, col2 = st.columns(2)

# KNN Model Column
with col1:
    st.markdown("<h3 style='text-align: center;'>KNN Model</h3>", 
                unsafe_allow_html=True)
    
    n_neighbors = st.number_input("Chọn n_neighbors (KNN)", 
                                min_value=1, max_value=20, value=4)
    selected_weights = st.selectbox("Chọn weights (KNN)", 
                                options=WEIGHTS_OPTIONS, index=1)
    selected_metrics_knn = st.selectbox("Chọn metrics (KNN)", 
                                    options=METRICS_OPTIONS, index=1)
    
    model_KNN = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights_map[selected_weights],
        metric=metrics_map[selected_metrics_knn]
    )
    
    model_KNN.fit(model_data['train_features'], 
                    model_data['train_labels_encoded'])
    y_pred_knn = model_KNN.predict(model_data['test_features'])
    
    Visualizer.plot_classification_report(
        model_data['test_labels_encoded'], 
        y_pred_knn, 
        model_data['label_encoder'].classes_, 
        "KNN"
    )
    
    Visualizer.plot_confusion_matrix(
        confusion_matrix(model_data['test_labels_encoded'], y_pred_knn),
        "KNN",
        model_data['label_encoder']
    )

# Custom Kernel SVM Column
with col2:
    st.markdown("<h3 style='text-align: center;'>Custom Kernel SVM</h3>", 
                unsafe_allow_html=True)
    
    selected_metrics_svm = st.selectbox("Chọn metric cho kernel", 
                                        options=METRICS_OPTIONS, 
                                        key='svm_m')
    C = st.number_input("Chọn C (regularization parameter)", 
                        min_value=0.1, max_value=10.0, 
                        value=1.0, step=0.1)
    
    custom_kernel = CustomKernelSVM.create_kernel(
        metrics_map[selected_metrics_svm]
    )
    
    model_SVM = SVC(kernel='precomputed', C=C)
    
    # Train SVM
    gram_matrix_train = custom_kernel(model_data['train_features'])
    model_SVM.fit(gram_matrix_train, model_data['train_labels_encoded'])
    
    # Test SVM
    gram_matrix_test = custom_kernel(
        model_data['test_features'], 
        model_data['train_features']
    )
    y_pred_svm = model_SVM.predict(gram_matrix_test)
    
    Visualizer.plot_classification_report(
        model_data['test_labels_encoded'], 
        y_pred_svm, 
        model_data['label_encoder'].classes_, 
        "Custom Kernel SVM"
    )
    
    Visualizer.plot_confusion_matrix(
        confusion_matrix(model_data['test_labels_encoded'], y_pred_svm),
        "Custom Kernel SVM",
        model_data['label_encoder']
    )

# Image upload and prediction section
st.markdown("<h3 style='text-align: center;'>Thử nghiệm</h3>", 
            unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Tải các hình ảnh lên",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    num_cols = min(len(uploaded_files), 10)
    cols = st.columns(num_cols)
    
    for i, uploaded_file in enumerate(uploaded_files):
        col = cols[i % num_cols]
        with col:
            # Process image
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, width=128)
            
            # Extract features
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (128, 128))
            image_features = FeatureExtractor.extract_features([img_resized])
            
            # Make predictions
            pred_knn = model_KNN.predict(image_features)[0]
            pred_svm = model_SVM.predict(
                custom_kernel(image_features, model_data['train_features'])
            )[0]
            
            # Display results
            caption = f"""
            <div style='text-align: center; color: black; margin-top: -10px;'>
                KNN: {MAPPING.get(model_data['label_encoder'].classes_[pred_knn], 'Unknown')}<br>
                SVM: {MAPPING.get(model_data['label_encoder'].classes_[pred_svm], 'Unknown')}
            </div>
            """
            st.markdown(caption, unsafe_allow_html=True)
