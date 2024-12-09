import pandas as pd
from joblib import load

def predict_do(npm):

    npm=int(npm)

    data = _prepare_data()

    model = load('./dropout-predict/do-predict.joblib')
    data_mhs = data[data['npm'] == npm]

    if data_mhs.empty:
        return f"Data {npm} tidak ditemukan."

    features = data_mhs[
        ['ipk_mahasiswa', 'attendance_rate', 'total_activity_points', 'activity_count', 'angkatan_mahasiswa']].values

    prediction = model.predict(features)

    # Map the prediction to a human-readable status (assuming binary classification)
    status_mapping = {0: 'Kemungkinan Rendah', 1: 'Drop-out (putus studi)'}
    predicted_status = status_mapping.get(prediction[0], 'Status tidak diketahui')

    return predicted_status

def _prepare_data():
    mhs = pd.read_csv('./MHS.csv')
    mhs = mhs.drop(columns=['pembimbing_tugas_akhir'])

    sa = pd.read_csv('./SA.csv')
    krs = pd.read_csv('./KRS.csv')

    point_mapping = {
        'Lokal': 1,
        'Provinsi': 3,
        'Nasional': 5,
        'International': 10
    }

    # Apply mapping
    sa['points'] = sa['tingkat_kegiatan'].str.lower().map(point_mapping).fillna(1)

    # Calculate total points for each student
    sa_points = sa.groupby('npm')['points'].sum().reset_index()
    sa_points.rename(columns={'points': 'total_activity_points'}, inplace=True)

    # Calculate attendance percentage per course
    krs['attendance_percentage'] = (krs['total_hadir'] / krs['total_pertemuan']) * 100

    total_sks = krs.groupby('npm')['sks_matakuliah'].sum().reset_index()
    total_sks.rename(columns={'sks_matakuliah': 'total_sks'}, inplace=True)

    attendance = krs.groupby('npm')['attendance_percentage'].mean().reset_index()
    attendance.rename(columns={'attendance_percentage': 'attendance_rate'}, inplace=True)

    # Merge SA and KRS aggregated data with MHS.csv

    data = mhs.merge(sa_points, on='npm', how='left')
    data = data.merge(total_sks, on='npm', how='left')
    data = data.merge(attendance, on='npm', how='left')

    data.fillna(0, inplace=True)

    # Fill missing values (e.g., students with no activities or KRS data)
    data['attendance_rate'] = data['attendance_rate'].fillna(0)
    data['ipk_mahasiswa'] = data['ipk_mahasiswa'].fillna(0)

    data.dropna(inplace=True)

    # data['attendance_rate'] = data['total_hadir'] / data['total_pertemuan']
    data['activity_count'] = sa.groupby('npm')['nama_kegiatan'].transform('count')
    data['activity_count'] = data['activity_count'].fillna(0)

    X = data[['ipk_mahasiswa', 'attendance_rate', 'total_activity_points','activity_count','angkatan_mahasiswa']]
    
    return data