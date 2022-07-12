from google_drive_downloader import GoogleDriveDownloader as gdd
import segmentation_models_pytorch as smp


def download_model():
    gdd.download_file_from_google_drive(file_id='1VcmNGuhh5QbiJXITxnd299c1WUv2oMQ9',
                                        dest_path='./pixel_wise_encoder.pt', showsize=True)


if __name__ == '__main__':
    download_model()
