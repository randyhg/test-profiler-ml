import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

def predict_activity_preference(npm):

    npm=int(npm)

    data = _prepare_data()
    # if not data:
    #     return None

    kmeans_model = load('./activity-preference/activity-preference-model.joblib')
    if not kmeans_model:
        return ""

    data['cluster'] = kmeans_model.labels_

    student_data = data[data['npm'] == npm]

    if student_data.empty:
        return ""

    #Choose features
    features = student_data[
        ['ipk_mahasiswa'] + list(student_data.columns[student_data.columns.str.contains('tingkat_kegiatan_')])]

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # Use the fitted scaler from the training phase

    # Predict the cluster for the student
    cluster = kmeans_model.predict(features_scaled)[0]

    # Get all data for students in the same cluster
    cluster_data = data[data['cluster'] == cluster]

    # Find the most common one-hot encoded column for tingkat_kegiatan in the cluster
    tingkat_kegiatan_cols = cluster_data.columns[cluster_data.columns.str.contains('tingkat_kegiatan_')]

    # Get the sum of each one-hot encoded column in the cluster
    counts = cluster_data[tingkat_kegiatan_cols].sum()

    # Identify the most common one
    favorite_tingkat_kegiatan_col = counts.idxmax()

    # Extract the original category name from the column name
    favorite_tingkat_kegiatan = favorite_tingkat_kegiatan_col.replace('tingkat_kegiatan_', '')

    return favorite_tingkat_kegiatan.capitalize()


def _prepare_data():
    data_mahasiswa = pd.read_csv('./MHS.csv')
    data_kegiatan = pd.read_csv("./SA.csv")

    merged_data = pd.merge(data_mahasiswa, data_kegiatan, on="npm")

    merged_data.dropna(inplace=True)

    data = merged_data[['npm', 'nama_mahasiswa', 'ipk_mahasiswa', 'nama_kegiatan', 'tingkat_kegiatan']]

    # One-hot encode tingkat_kegiatan
    data = pd.get_dummies(data, columns=['tingkat_kegiatan'])

    return data