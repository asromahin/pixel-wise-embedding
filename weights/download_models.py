import gdown
import segmentation_models_pytorch as smp


def download_model():
    # gdd.download_file_from_google_drive(file_id='1VcmNGuhh5QbiJXITxnd299c1WUv2oMQ9',
    #                                     dest_path='./pixel_wise_encoder_download.pt', showsize=True, unzip=False)
    url = 'https://drive.google.com/file/d/1JDBF6FAjVUNRM83vsVaTjVMf3W1xFmMM/view?usp=sharing'
    # id = "1JDBF6FAjVUNRM83vsVaTjVMf3W1xFmMM"
    output = 'pixel_wise_encoder_download.pt'
    gdown.download(url=url, output=output, quiet=False, fuzzy=True, use_cookies=False)


if __name__ == '__main__':
    download_model()
