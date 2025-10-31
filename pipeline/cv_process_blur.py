from common.elements.legacy.dataset import get_dataset_info, DatasetInfo, Dataset
from common.elements.visualize import create_tb, delete_tb
from elements.load_data.load_data import load_image
from elements.predict.predict import blur
from elements.save_results.save_results import save_image

filename =  get_dataset_info(Dataset.POTATO_PLANT_TILES, DatasetInfo.PREVIEW)
output_filename = "/media/public_data/temp/Nirvana/image1.png"
print(filename)

writer = create_tb("tb")

img = load_image(filename)
img_blurred = blur(img)
save_image(img_blurred, output_filename)
input("please Enter to continue...")
delete_tb("tb")
