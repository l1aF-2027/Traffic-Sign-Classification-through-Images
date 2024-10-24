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

# [Giữ nguyên các hàm helper từ code gốc]
def color_histogram(image):
    row, column, channel = image.shape[:3]
    size = row * column
    feature = []
    for k in range(channel):
        histogram = np.squeeze(cv2.calcHist([image], [k], None, [8], [0, 256]))
        histogram = histogram / size
        feature.extend(histogram)
    return feature

def hog(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_features = skimage_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys', transform_sqrt=True)
    return hog_features

def extract_features(images):
    color_features = [color_histogram(image) for image in images]
    hog_features = [hog(image) for image in images]
    combined_features = [np.concatenate((color_feature, hog_feature)) for color_feature, hog_feature in zip(color_features, hog_features)]
    return combined_features

def chi_square_distance(x, y):
    return cv2.compareHist(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), cv2.HISTCMP_CHISQR)

def correlation_distance(x, y):
    return 1 - cv2.compareHist(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), cv2.HISTCMP_CORREL)

def bhattacharyya_distance(x, y):
    return cv2.compareHist(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), cv2.HISTCMP_BHATTACHARYYA)

def intersection_distance(x, y):
    return 1 - cv2.compareHist(np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), cv2.HISTCMP_INTERSECT)

def euclidean_distance(x, y):
    return np.linalg.norm(np.array(x, dtype=np.float32) - np.array(y, dtype=np.float32))

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

def plot_decision_boundaries(X, y, model, model_name, pca):
    """Vẽ decision boundaries của model"""
    st.markdown(f"<h6 style='text-align: left;'>Decision Boundaries - {model_name}</h6>", unsafe_allow_html=True)
    
    # Tạo lưới điểm để vẽ decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Dự đoán cho mỗi điểm trong lưới
    if model_name == 'Custom Kernel SVM':
        # Đối với SVM với custom kernel, cần tính gram matrix
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        # Inverse transform để lấy dữ liệu về không gian gốc
        grid_points_original = pca.inverse_transform(grid_points)
        # Tính gram matrix
        gram_matrix_grid = create_custom_kernel(map_metrics.get(selected_metrics_svm))(
            grid_points_original, 
            train_features
        )
        Z = model.predict(gram_matrix_grid)
    else:
        # Đối với KNN
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        # Inverse transform để lấy dữ liệu về không gian gốc
        grid_points_original = pca.inverse_transform(grid_points)
        Z = model.predict(grid_points_original)
    
    Z = Z.reshape(xx.shape)
    
    # Vẽ decision boundary
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.4)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.colorbar(scatter)
    
    plt.title(f'Decision Boundaries - {model_name}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    # Thêm legend
    classes = label_encoder.classes_
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=scatter.cmap(scatter.norm(i)),
                                label=classes[i], markersize=10)
                      for i in range(len(classes))]
    ax.legend(handles=legend_elements)
    
    st.pyplot(fig)

def plot_pca_variance(pca):
    """Vẽ biểu đồ explained variance ratio của PCA"""
    st.markdown("<h6 style='text-align: left;'>PCA Explained Variance Ratio</h6>", unsafe_allow_html=True)
    
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


def create_custom_kernel(metric_func):
    def custom_kernel(X, Y=None):
        if Y is None:
            Y = X
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                # Chuyển đổi khoảng cách thành độ tương tự bằng cách lấy nghịch đảo
                similarity = 1.0 / (1.0 + metric_func(X[i], Y[j]))
                gram_matrix[i, j] = similarity
        return gram_matrix
    return custom_kernel

# Cấu hình trang
st.set_page_config(
    page_title="Traffic Sign Classification Web",
    page_icon=":vertical_traffic_light:",
    layout="wide"  # Thêm layout wide để có nhiều không gian hơn cho 2 cột
)

st.markdown("<h1 style='text-align: center;'>Dự đoán biển báo từ hình ảnh</h1>", unsafe_allow_html=True)

# Load models và data
path_joblib = r'joblib/'
model = joblib.load(path_joblib + 'best_knn_model.joblib')
label_encoder = joblib.load(path_joblib + 'label_encoder.joblib')
train_features = joblib.load(path_joblib + 'train_features.joblib')
test_features = joblib.load(path_joblib + 'test_features.joblib')
train_labels_encoded = joblib.load(path_joblib + 'train_labels_encoded.joblib')
test_labels_encoded = joblib.load(path_joblib + 'test_labels_encoded.joblib')

# Chia layout thành 2 cột
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3 style='text-align: center;'>KNN Model</h3>", unsafe_allow_html=True)
    
    weights_options = ['Uniform', 'Distance']
    metrics_options = ['Chi-Square', 'Correlation', 'Bhattacharyya', 'Intersection', 'Euclidean']
    
    map_metrics = {
        'Chi-Square': chi_square_distance,
        'Correlation': correlation_distance,
        'Bhattacharyya': bhattacharyya_distance,
        'Intersection': intersection_distance,
        'Euclidean': euclidean_distance
    }
    
    map_weights = {
        'Uniform': 'uniform',
        'Distance': 'distance'
    }
    
    n_neighbors = st.number_input("Chọn n_neighbors (KNN)", min_value=1, max_value=20, value=4, key='knn_n')
    selected_weights = st.selectbox("Chọn weights (KNN)", options=weights_options, index=1, key='knn_w')
    selected_metrics_knn = st.selectbox("Chọn metrics (KNN)", options=metrics_options, index=1, key='knn_m')
    
    model_KNN = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=map_weights.get(selected_weights),
        metric=map_metrics.get(selected_metrics_knn)
    )
    
    model_KNN.fit(train_features, train_labels_encoded)
    y_pred_knn = model_KNN.predict(test_features)
    
    plot_classification_report(test_labels_encoded, y_pred_knn, label_encoder.classes_, "KNN")
    plot_cm(confusion_matrix(test_labels_encoded, y_pred_knn), "KNN")

# Cột 2: Custom Kernel SVM
with col2:
    st.markdown("<h3 style='text-align: center;'>Custom Kernel SVM</h3>", unsafe_allow_html=True)
    
    selected_metrics_svm = st.selectbox("Chọn metric cho kernel", options=metrics_options, key='svm_m')
    C = st.number_input("Chọn C (regularization parameter)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key='svm_c')
    
    # Tạo custom kernel từ metric đã chọn
    custom_kernel_func = create_custom_kernel(map_metrics.get(selected_metrics_svm))
    
    model_SVM = SVC(kernel='precomputed', C=C)
    
    # Tính gram matrix cho training
    gram_matrix_train = custom_kernel_func(train_features)
    
    # Fit model
    model_SVM.fit(gram_matrix_train, train_labels_encoded)
    
    # Tính gram matrix cho testing
    gram_matrix_test = custom_kernel_func(test_features, train_features)
    
    # Predict
    y_pred_svm = model_SVM.predict(gram_matrix_test)
    
    plot_classification_report(test_labels_encoded, y_pred_svm, label_encoder.classes_, "Custom Kernel SVM")
    plot_cm(confusion_matrix(test_labels_encoded, y_pred_svm), "Custom Kernel SVM")
    

st.markdown("<h3 style='text-align: center;'>Model Visualization</h3>", unsafe_allow_html=True)

# Thực hiện PCA
n_components = st.slider("Số components cho PCA", min_value=2, max_value=min(50, len(train_features[0])), value=2)
pca = PCA(n_components=n_components)

# Transform dữ liệu training
X_train_pca = pca.fit_transform(train_features)
X_test_pca = pca.transform(test_features)

# Vẽ biểu đồ explained variance
plot_pca_variance(pca)

# Nếu chọn 2 components, vẽ decision boundaries
if n_components == 2:
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Train và vẽ KNN
        model_KNN_pca = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=map_weights.get(selected_weights),
            metric=map_metrics.get(selected_metrics_knn)
        )
        model_KNN_pca.fit(X_train_pca, train_labels_encoded)
        plot_decision_boundaries(X_train_pca, train_labels_encoded, 
                               model_KNN_pca, "KNN", pca)
    
    with viz_col2:
        # Train và vẽ SVM
        custom_kernel_func = create_custom_kernel(map_metrics.get(selected_metrics_svm))
        gram_matrix_train = custom_kernel_func(train_features)
        model_SVM.fit(gram_matrix_train, train_labels_encoded)
        plot_decision_boundaries(X_train_pca, train_labels_encoded, 
                               model_SVM, "Custom Kernel SVM", pca)

# Thêm scatter plot 3D nếu chọn 3 components
elif n_components == 3:
    st.markdown("<h6 style='text-align: left;'>3D Scatter Plot của 3 Principal Components đầu tiên</h6>", 
                unsafe_allow_html=True)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2],
                        c=train_labels_encoded, cmap='viridis')
    
    plt.colorbar(scatter)
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    
    # Thêm legend
    classes = label_encoder.classes_
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=scatter.cmap(scatter.norm(i)),
                                label=classes[i], markersize=10)
                      for i in range(len(classes))]
    ax.legend(handles=legend_elements)
    
    st.pyplot(fig)

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
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (128, 128))
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