#  my main script for silhouettes + simple measurements
#  goes through new images and old images and makes masks, line images, a csv with 3 widths

import cv2
import numpy as np
import os
def make_gray(img):

    # manual rgb to gray (from class)
    R = img[:,:,2].astype(float)
    G = img[:,:,1].astype(float)
    B = img[:,:,0].astype(float)
    g = 0.2989*R + 0.5871*G + 0.1140*B
    return g.astype(np.uint8)


def stretch_gray(g):

    #min max contrast stretch
    gmin = g.min()
    gmax = g.max()
    if gmin == gmax:
        return g
    out = (g - gmin) * (255.0 / (gmax - gmin))
    return out.astype(np.uint8)


def biggest_blob(mask):

    num_labels, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return mask
    # skip background 
    areas = stats[1:,4]
    best_idx = 1 + np.argmax(areas)
    out = np.uint8(labels == best_idx) * 255
    return out


def make_silhouette(img):

    g = make_gray(img)
    g2 = stretch_gray(g)
    g2 = cv2.GaussianBlur(g2,(5,5),1.0)

    # otsu so foreground is white
    T,mask = cv2.threshold(g2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    k = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,k,1)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,1)

    mask = biggest_blob(mask)
    return mask




def measure_and_lines(mask):
    #25%,50%,75% height widths + line image
    m = np.uint8(mask > 0) * 255
    ys, xs = np.where(m>0)
    if len(ys)==0:
        # no silhouette
        vis = cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)
        return (0,0,0), vis


    y_top = ys.min()
    y_bot = ys.max()
    H = y_bot - y_top

    vis = cv2.cvtColor(m,cv2.COLOR_GRAY2BGR)

# draw lines at 25%, 50%, 75%
    def one_line(frac, color):
        y = int(y_top + frac*H)
        if y<0 or y>=m.shape[0]:
            return 0
        row = np.where(m[y,:] > 0)[0]
        if len(row)==0:
            return 0
        x1 = int(row.min())
        x2 = int(row.max())
        cv2.line(vis,(x1,y),(x2,y),color,2)
        return (x2-x1)

    w25 = one_line(0.25,(255,0,0))
    w50 = one_line(0.50,(0,255,0))
    w75 = one_line(0.75,(0,0,255))

    return (int(w25),int(w50),int(w75)), vis


# process one folder of images

def process_folder(label, folder_path, results_root):

    print()
    print(f"{label} images")

    out_masks = os.path.join(results_root, label + "_masks")
    out_lines = os.path.join(results_root, label + "_lines")

    os.makedirs(out_masks, exist_ok=True)
    os.makedirs(out_lines, exist_ok=True)

    csv_path = os.path.join(results_root, label + "_measurements.csv")
    f = open(csv_path,"w")
    f.write("filename,w25,w50,w75\n")


    valid_ext = (
       ".jpg", ".jpeg", ".webp", ".avif", ".pjpeg", ".jfif"
    )

# process each image
    for name in sorted(os.listdir(folder_path)):
        if not name.lower().endswith(valid_ext):
            continue

        ipath = os.path.join(folder_path, name)
        print(name)

        img = cv2.imread(ipath)
        if img is None:
            continue

        sil = make_silhouette(img)

        base, _ = os.path.splitext(name)
        mask_out = os.path.join(out_masks, base + "_mask.png")
        cv2.imwrite(mask_out, sil)

        (w25, w50, w75), vis = measure_and_lines(sil)
        lines_out = os.path.join(out_lines, base + "_lines.png")
        cv2.imwrite(lines_out, vis)

        f.write(f"{name},{w25},{w50},{w75}\n")

    f.close()
    print("csv:", csv_path)


def main():

    # paths to input and output folders
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    new_dir = os.path.join(root, "new_images")
    old_dir = os.path.join(root, "old_images")
    res_dir = os.path.join(root, "results")
    os.makedirs(res_dir, exist_ok=True)

# process both folders old and new
    if os.path.isdir(new_dir):
        process_folder("new", new_dir, res_dir)
    else:
        print("no new_images")

    if os.path.isdir(old_dir):
        process_folder("old", old_dir, res_dir)
    else:
        print("no old_images")

    print()
    print("done")


if __name__ == "__main__":
    main()
