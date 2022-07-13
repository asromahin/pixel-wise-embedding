from google_drive_downloader import GoogleDriveDownloader as gdd
import segmentation_models_pytorch as smp


def download_model():
    # gdd.download_file_from_google_drive(file_id='1hEnPGKYM0k3QO5WvLUSoEula8rs9tG4J',
    #                                     dest_path='./pixel_wise_encoder_download.zip', unzip=False)
    gdd.download_file_from_google_drive(
        "1snnIkz3eUQ8_cFVJ4Ui1ekLmNBq5v8N6",
        dest_path='./pixel_wise_encoder_download.zip',
        overwrite=False,
        unzip=True,
        showsize=True,
    )
# https://drive.google.com/file/d/1hEnPGKYM0k3QO5WvLUSoEula8rs9tG4J/view?usp=sharing

if __name__ == '__main__':
    download_model()
