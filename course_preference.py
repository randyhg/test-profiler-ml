import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder

def get_favorite_courses(npm):  
    # convert to int
    npm = int(npm)

    dataset, scaler = _prepare_data()

    kmeans_model = load('./course-preference/course-preference-model-fixed.joblib')
    if not kmeans_model:
        return ""

    dataset['cluster'] = kmeans_model.labels_
    
    # get the student data using npm
    student_data = dataset[dataset['npm'] == npm]
    
    if student_data.empty:  
        return "" 
    
    # take the appropriate features to make predictions
    features = student_data[['total_hadir', 'sks_matakuliah', 'kode_nilai_numerik', 'kategori_matakuliah_encoded']]  
    features_scaled = scaler.transform(features)

    # predict the cluster 
    cluster = kmeans_model.predict(features_scaled)  

    # make sure the 'cluster' column is present in dataset 
    favorite_courses = dataset[(dataset['cluster'] == cluster[0]) & (dataset['npm'] == npm)]

    result = favorite_courses[['nama_mahasiswa', 'kode_matkul', 'kategori_matakuliah']].drop_duplicates(subset=['kategori_matakuliah'])  

    if not result.empty:
        return result['kategori_matakuliah'].iloc[0]
    else:
        return ""

def _prepare_data():
    data_mahasiswa = pd.read_csv('./MHS.csv')
    data_krs = pd.read_csv('./KRS.csv')

    # Merge data_krs and data_mahasiswa using 'npm' as the key
    merged_data = pd.merge(data_krs, data_mahasiswa, on='npm')

    # Drop null data
    merged_data.dropna(inplace=True)

    # Apply conditions directly to merged_data
    merged_data = merged_data[
        (merged_data['total_terlaksana'] != 0) &  # Condition 1: total_terlaksana is not 0
        ((merged_data['total_hadir'] != 0) | (merged_data['kode_nilai'].isna())) &  # Condition 2: total_hadir is not 0 OR kode_nilai is NaN
        (merged_data['total_pertemuan'] != 0)  # Condition 3: total_pertemuan is not 0
    ]

    # Optionally, drop any remaining NaN values if needed
    merged_data.dropna(inplace=True)

    # mapping `nilai`
    nilai_mapping = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0}  
    merged_data['kode_nilai_numerik'] = merged_data['kode_nilai'].map(nilai_mapping)  


    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Apply label encoding to the 'kategori_matakuliah' column
    merged_data['kategori_matakuliah_encoded'] = label_encoder.fit_transform(merged_data['kategori_matakuliah'])

    # Now, use the encoded 'kategori_matakuliah' for clustering
    features = merged_data[['total_hadir', 'sks_matakuliah', 'kode_nilai_numerik', 'kategori_matakuliah_encoded']]
    
    # normalize data using loaded scaler
    scaler = load('./activity-preference/scaler.joblib')
    features_scaled = scaler.fit_transform(features)  

    pd.DataFrame(features_scaled, columns=features.columns)
    
    return merged_data, scaler