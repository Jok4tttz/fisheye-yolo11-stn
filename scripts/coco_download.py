from pathlib import Path
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

# Download labels
segments = False  # segment or box labels
dir = Path("./datasources/COCO2017")  # dataset root dir
urls = [ASSETS_URL + ("/coco2017labels-segments.zip" if segments else "/coco2017labels.zip")]  # labels
download(urls, dir=dir)
# Download data
# urls = [
#     "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
#     "http://images.cocodataset.org/zips/val2017.zip"  # 1G, 5k images
# ]
# download(urls, dir=dir / "images", threads=3)