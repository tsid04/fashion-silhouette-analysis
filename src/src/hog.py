# will run HOG on all my images at once in new images and old images

import os
import cv2

from skimage.feature import hog
from skimage import exposure



def do_hog(gray_img):
    # resize so everything is same size
    small = cv2.resize(gray_img,(128,128))

    feats, hog_img = hog(
        small,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        visualize=True
    )

# rescale to 0-255
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, hog_img.max()))
    hog_img = (hog_img * 255).astype("uint8")
    return feats, hog_img


def run_folder(label, folder_path, results_root):
    print()
    print(f"{label} images:")


    out_dir = os.path.join(results_root, label + "_hog")
    os.makedirs(out_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".webp", ".avif", ".pjpeg", ".jfif")
# go through each image in the folder
for fname in sorted(os.listdir(folder_path)):
    if not fname.lower().endswith(exts):
        continue  #skip any non images

    ipath = os.path.join(folder_path, fname)
    print(f"   processing {fname}...")

    g = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
    if g is None:
        print("      (couldn't read image, skipping")
        continue

    feats, himg = do_hog(g)
    print(f"      HOG feature length: {len(feats)}")

    base, _ = os.path.splitext(fname)
    opath = os.path.join(out_dir, base + "_hog.png")
    cv2.imwrite(opath, himg)



def main():
    # paths to input and output folders
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    new_dir = os.path.join(root, "new_images")
    old_dir = os.path.join(root, "old_images")
    res_dir = os.path.join(root, "results")
    os.makedirs(res_dir, exist_ok=True)

    os.makedirs(res_dir, exist_ok=True)
    
# process both folders old and new
    if os.path.isdir(new_dir):
        run_folder("new", new_dir, res_dir)
    else:
        print("no new_images folder")

    if os.path.isdir(old_dir):
        run_folder("old", old_dir, res_dir)
    else:
        print("no old_images folder")

    print("")
    print("hog done")


if __name__=="__main__":
    main()
