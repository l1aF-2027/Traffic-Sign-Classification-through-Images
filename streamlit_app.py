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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import cityblock, cosine, correlation, sqeuclidean, euclidean
import os

project_dir = os.getcwd()

# [Giữ nguyên các hàm helper từ code gốc]
def blur_image(image):
    blurred_image = cv2.medianBlur(image, 5)
    return blurred_image

def color_histogram(image):
    row, column, channel = image.shape[:3]
    size = row * column
    feature = []
    for k in range(channel):
        histogram = np.squeeze(cv2.calcHist([image], [k], None, [32], [0, 256]))
        histogram = histogram / size
        feature.extend(histogram)
    return feature

def hog(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_features = skimage_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys', transform_sqrt=True)
    return hog_features

def extract_features(images):
    blurred_images = [blur_image(image) for image in images]
    color_features = [color_histogram(image) for image in blurred_images]
    hog_features = [hog(image) for image in blurred_images]
    combined_features = [np.concatenate((color_feature, hog_feature)) for color_feature, hog_feature in zip(color_features, hog_features)]
    return combined_features

def chi_square_distance(x, y):
    return cv2.compareHist(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), cv2.HISTCMP_CHISQR)

def bhattacharyya_distance(x, y):
    return cv2.compareHist(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), cv2.HISTCMP_BHATTACHARYYA)

def intersection_distance(x, y):
    return 1 - cv2.compareHist(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), cv2.HISTCMP_INTERSECT)

def plot_cm(cm, model_name):
    st.markdown(f"<h6 style='text-align: left;'>Confusion Matrix - {model_name}</h6>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', pad=20)
    plt.colorbar(cax)
    ax.set_xticklabels([''] + list(label_encoder.classes_), rotation=45)
    ax.set_yticklabels([''] + list(label_encoder.classes_))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), val in np.ndenumerate(cm):
        text_color = 'white' if cm[i, j] > np.max(cm)/2 else 'black'
        ax.text(j, i, val, ha='center', va='center', color=text_color)
    st.pyplot(fig)

def plot_classification_report(y_true, y_pred, labels, model_name):
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df_report = pd.DataFrame(report).T
    df_report = df_report.round(3)
    df_report = df_report.drop(['support'], axis=1)
    
    st.markdown(f"<h6 style='text-align: left;'>Classification Report - {model_name}</h6>", unsafe_allow_html=True)
    st.dataframe(df_report)

# Cấu hình trang
st.set_page_config(
    page_title="Traffic Sign Classification Web",
    page_icon=":vertical_traffic_light:",
    layout="wide"  # Thêm layout wide để có nhiều không gian hơn cho 2 cột
)

st.markdown("<h1 style='text-align: center;'>Dự đoán biển báo từ hình ảnh</h1>", unsafe_allow_html=True)

# Load models và data

model_knn = joblib.load(project_dir + '/joblib/best_knn_model.joblib')
model_svm = joblib.load(project_dir + '/joblib/best_svm_model.joblib')
label_encoder = joblib.load(project_dir + '/joblib/label_encoder.joblib')
train_features = joblib.load(project_dir + '/joblib/train_features.joblib')
test_features = joblib.load(project_dir + '/joblib/test_features.joblib')
train_labels_encoded = joblib.load(project_dir + '/joblib/train_labels_encoded.joblib')
test_labels_encoded = joblib.load(project_dir + '/joblib/test_labels_encoded.joblib')

# Chia layout thành 2 cột
col1, col2 = st.columns(2)

# Cột 1: KNN Model
with col1:
    st.markdown("<h3 style='text-align: center;'>KNN Model</h3>", unsafe_allow_html=True)

    best_model_knn = st.checkbox("Sử dụng Best KNN Model", value=True)

    weights_options = ['Uniform', 'Distance']
    metrics_options = [
        'cityblock',
        'cosine',
        'correlation',
        'euclidean',
        'sqeuclidean',
        'chi_square',
        'bhattacharyya',
        'intersection'
    ]
    
    map_metrics = {
        'cityblock': cityblock,
        'cosine': cosine,
        'correlation': correlation,
        'euclidean': euclidean,
        'sqeuclidean': sqeuclidean,
        'chi_square': chi_square_distance,
        'bhattacharyya': bhattacharyya_distance,
        'intersection': intersection_distance
    }
    
    map_weights = {
        'Uniform': 'uniform',
        'Distance': 'distance'
    }

    if not best_model_knn:
        n_neighbors = st.number_input("Chọn n_neighbors", min_value=1, max_value=20, value=4)
        selected_weights = st.selectbox("Chọn weights", options=weights_options, index=1)
        selected_metrics = st.selectbox("Chọn metrics", options=metrics_options, index=1)
        
        leaf_size_options = [10, 20, 30, 40, 50]
        leaf_size = st.selectbox("Chọn leaf_size", options=leaf_size_options, index=1)
        
        model_KNN = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=map_weights.get(selected_weights),
            metric=map_metrics.get(selected_metrics),
            leaf_size=leaf_size
        )
        
        model_KNN.fit(train_features, train_labels_encoded)
        y_pred_knn = model_KNN.predict(test_features)

    else:
        model_KNN = model_knn
        y_pred_knn = model_KNN.predict(test_features)

    plot_classification_report(test_labels_encoded, y_pred_knn, label_encoder.classes_, "KNN")
    plot_cm(confusion_matrix(test_labels_encoded, y_pred_knn), "KNN")

# Cột 2: SVM Model
with col2:
    st.markdown("<h3 style='text-align: center;'>SVM Model</h3>", unsafe_allow_html=True)
    best_model_svm  = st.checkbox("Sử dụng Best SVM Model", value=True)
    
    kernel_options = ['linear', 'rbf', 'poly']
    if not best_model_knn:
        selected_kernel = st.selectbox("Chọn kernel", options=kernel_options, index=1)
        C = st.number_input("Chọn C (regularization parameter)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    
        model_SVM = SVC(kernel=selected_kernel, C=C)
        model_SVM.fit(train_features, train_labels_encoded)
        y_pred_svm = model_SVM.predict(test_features)
    else:
        model_SVM = model_svm
        y_pred_svm = model_SVM.predict(test_features)

    
    plot_classification_report(test_labels_encoded, y_pred_svm, label_encoder.classes_, "SVM")
    plot_cm(confusion_matrix(test_labels_encoded, y_pred_svm), "SVM")

# Phần thử nghiệm (full width)
st.markdown("<h3 style='text-align: center;'>Thử nghiệm</h3>", unsafe_allow_html=True)

mapping = {
    'Nguyhiem': 'Nguy hiểm',
    'Cam': 'Cấm',
    'Chidan': 'Chỉ dẫn',
    'Hieulenh': 'Hiệu lệnh',
    'Phu': 'Phụ'
}

uploaded_files = st.file_uploader("Tải các hình ảnh lên", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images.append(image)

    num_cols = 10
    cols = st.columns(num_cols)
    
    for i, img in enumerate(images):
        col = cols[i % num_cols]
        with col:
            st.image(img, use_column_width=True, width=128)
            
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_bgr, (64, 64))
            image_inputs = extract_features([img_resized])
            
            # Dự đoán từ cả hai model
            pred_knn = model_KNN.predict(image_inputs)[0]
            pred_svm = model_SVM.predict(image_inputs)[0]
            
            caption = f"""
            <div style='text-align: center; color: black; margin-top: -10px;'>
                KNN: {mapping.get(label_encoder.classes_[pred_knn], 'Unknown')}<br>
                SVM: {mapping.get(label_encoder.classes_[pred_svm], 'Unknown')}
            </div>
            """
            st.markdown(caption, unsafe_allow_html=True)