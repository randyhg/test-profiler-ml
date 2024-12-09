import pandas as pd

from activity_preference import predict_activity_preference
from course_preference import get_favorite_courses
from dropout_predict import predict_do

def get_mahasiswa_info(npm):
    df = pd.read_csv('./MHS.csv')
    mhs_info = df[df['npm'] == npm]

    if mhs_info.empty:
        return None
    
    return mhs_info.iloc[0]

def test_functions(npm):
    mhs = get_mahasiswa_info(npm)
    if mhs is None:
        print(f"Mahasiswa dengan NPM {npm} tidak ditemukan.")
        return

    print("\n****************************************************")
    print(f"Nama: {mhs['nama_mahasiswa']}")
    print(f"Prodi: {mhs['prodi_mahasiswa']}")
    print(f"Status: {mhs['status_mahasiswa']}")


    # test predict_activity_preference
    try:
        activity_preference_result = predict_activity_preference(npm)
        print(f"Tingkat Kegiatan Favorit: {activity_preference_result}")
    except Exception as e:
        print(f"Error in predict_activity_preference for npm {npm}: {e}")

    # test get_favorite_courses
    try:
        favorite_courses_result = get_favorite_courses(npm)
        print(f"Matakuliah Favorit: {favorite_courses_result}")
    except Exception as e:
        print(f"Error in get_favorite_courses for npm {npm}: {e}")

    # test predict_do
    try:
        dropout_prediction_result = predict_do(npm)
        print(f"Persentase Dropout: {dropout_prediction_result}")
    except Exception as e:
        print(f"Error in predict_do for npm {npm}: {e}")
    
    print("****************************************************\n")

if __name__ == "__main__":
    test_npm = input("Silahkan input npm mahasiswa: ")

    test_npm = int(test_npm)

    test_functions(test_npm)