import zipfile
import shutil
import os

def move_file(source, destination):
    try:
        shutil.move(source, destination)
        print(f"파일 '{source}'가 '{destination}'로 이동되었습니다.")
    except FileNotFoundError:
        print(f"파일 '{source}'를 찾을 수 없습니다.")
    except PermissionError:
        print(f"이동할 권한이 없습니다.")
    except Exception as e:
        print(f"파일 이동 중 오류가 발생했습니다: {e}")

def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"파일 {file_path}이(가) 성공적으로 삭제되었습니다.")
    except FileNotFoundError:
        print(f"파일 {file_path}을(를) 찾을 수 없습니다.")
    except PermissionError:
        print(f"파일 {file_path}을(를) 삭제할 권한이 없습니다.")
    except Exception as e:
        print(f"파일 {file_path}을(를) 삭제하는 중 오류가 발생했습니다: {e}")

def zip_directory(directory, zip_file):
    """
    디렉토리를 압축 파일로 만드는 함수
    :param directory: 압축할 디렉토리 경로
    :param zip_file: 생성될 압축 파일 이름
    """
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                # 파일의 실제 경로
                file_path = os.path.join(root, file)
                # 압축 파일에 추가
                zipf.write(file_path, os.path.relpath(file_path, directory))

