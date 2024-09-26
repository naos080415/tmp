import cv2

import cv2
import numpy as np

from PIL import Image

from include.huggingface_cloth_segmentation.process import load_seg_model
from include.huggingface_cloth_segmentation.process import get_palette
from include.huggingface_cloth_segmentation.process import generate_mask

class FeatureMatching:
    '''
    2つの画像の特徴点を抽出し、特徴点のマッチングを行うクラス
    '''
    def __init__(self, method='akaze', inlier_num_threshold=10):
        self.method = method
        self.inlier_num_threshold = inlier_num_threshold

    def extract_features(self, image, threshold = 0.0000015):
        '''
        画像から特徴点を抽出
        '''
        if self.method == 'akaze':
            detector = cv2.AKAZE_create()
            detector.setThreshold(threshold)
        elif self.method == 'sift':
            detector = cv2.SIFT_create()
        elif self.method == 'surf':
            detector = cv2.SURF_create()
        elif self.method == 'orb':
            detector = cv2.ORB_create()
        else:
            raise ValueError('Invalid method')
        
        keypoints, descriptors = detector.detectAndCompute(image, None)

        return keypoints, descriptors
    
    def match_features(self, image1, keypoints1, descriptors1, image2, keypoints2, descriptors2):
        '''
        2つの画像の特徴点をマッチング
        '''
        found = False

        if self.method in ['akaze', 'sift', 'surf']:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.method == 'orb':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError('Invalid method')
        
        matches = matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        if 1:
            dist = [m.distance for m in matches]
            ret = sum(dist) / len(dist)
            print("ans : {}".format(ret))

        # 変換行列をRANSACで推定
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        transform_matrix, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)

        # 画像中のターゲットの中心を計算
        mat_center = np.array([[image2.shape[1] / 2] , [image2.shape[0] / 2], [1]])
        mat_center_inmap = transform_matrix @ mat_center

        mat_corners = np.array([[0, 0, 1], [0, image2.shape[0], 1], [image2.shape[1], image2.shape[0], 1], [image2.shape[1], 0, 1]])
        mat_corners_inmap = [transform_matrix @ corner for corner in mat_corners]


        pts = [tuple(map(int, corner.flatten())) for corner in mat_corners_inmap]
        pt_center = tuple(map(int, mat_center_inmap.flatten()[:2]))

        # 元画像にマッチング結果を描画
        if mask is not None and np.sum(mask) > 10:
            # line_color = (0, 0, 255)
            # for i in range(4):
            #     cv2.line(image1, pts[i], pts[(i + 1) % 4], line_color, 3)
            # cv2.circle(image1, pt_center, 5, line_color, cv2.FILLED)
            found = True
        matches_mask = mask.ravel().tolist()

        # 一致した特徴点のみを描画
        draw_params = dict(matchColor=(0, 255, 0),  # 一致した特徴点を緑で描画
                        singlePointColor=None,
                        matchesMask=matches_mask,  # RANSACで一致した特徴点のみ描画
                        flags=2)

        # 2画像間のマッチング結果画像を作成
        img1_2 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, **draw_params)
        # result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

        # decoded_bytes = cv2.imencode('.jpg', img1_2)[1].tobytes()
        # display(Image(data=decoded_bytes))
        return img1_2, found

    def imgsim(self, img0, img1):
        import imgsim
        import cv2

        vtr = imgsim.Vectorizer()
        print(img1)

        # img0 = cv2.imread("img0.png")
        # img1 = cv2.imread("img1.png")

        vec0 = vtr.vectorize(img0)
        vec1 = vtr.vectorize(img1)
        vecs = vtr.vectorize(np.array([img0, img1]))

        dist = imgsim.distance(vec0, vec1)
        print("distance =", dist)

        dist = imgsim.distance(vecs[0], vecs[1])
        print("distance =", dist)

cap = cv2.VideoCapture(0)
def get_image():
    ret, frame = cap.read()

    return frame

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def run_cloth_segmentation(img):

    device = 'cpu'
    # Create an instance of your model
    model = load_seg_model('model/cloth_seg.pth', device=device)
    palette = get_palette(4)
    cloth_seg = generate_mask(img, net=model, palette=palette, device=device)

def cloth_segmentation( cv_image):
    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    img = cv2pil(cv_image)
    run_cloth_segmentation(img=img)

    img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    cloth_color_list, out_dir = [], './output/alpha/*.png'
    import glob
    for mask_path in sorted(glob.glob(out_dir)):
        mask_img_gs = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)     # グレースケール画像

        # cloth_color = self.analyze_cloth_color(img_rgb=img_rgb, mask_img_gs=mask_img_gs)
        # cloth_color_list.append(cloth_color)

    for i in sorted(glob.glob(out_dir)):
        import os
        os.remove(i)
        pass

        
    clothes_mask_img_gs = cv2.imread('./output/cloth_seg/final_seg.png', cv2.IMREAD_GRAYSCALE)     # グレースケール画像

    def apply_mask(image, mask):
        result = cv2.bitwise_and(image, image, mask=mask)
        return result

    clothes_img = apply_mask(image=cv_image, mask=clothes_mask_img_gs)

    return clothes_img

cnt = 0
def hue_hist(img, bins=180):
    global cnt
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    saturation_channel = hsv[:, :, 1]
    color_mask = np.where(saturation_channel < 10, 0, 255).astype(np.uint8)     # 低彩度を無視したマスク画像
    low_saturation = np.where((saturation_channel > 0) & (saturation_channel <= 10), 255, 0).astype(np.uint8)
    sat_hist_num = np.count_nonzero(low_saturation)
    # print(sat_hist_num)


    hist = cv2.calcHist([hsv], [0], color_mask, [bins], [0, 180])
    new_element = np.array([sat_hist_num], dtype=hist.dtype)
    # print(hist.shape, sat_hist_num.shape, len(sat_hist_num))
    # print(type(hist), type(sat_hist_num))

    # print(hist)
    # hist = np.concatenate((hist, sat_hist_num))
    hist = np.append(hist, new_element)
    # print(type(hist[0]), type(hist[180]))
    print(len(hist))

    hist = cv2.normalize(hist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    total_sum = sum(hist)
    for i in range(len(hist)):
        hist[i] = hist[i] / total_sum

    # print(sat_hist_num + np.count_nonzero(color_mask), total_sum[0])
    return hist

def custom_hist(img, bins=180):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clothes_mask_img_gs = cv2.imread('./output/cloth_seg/final_seg.png', cv2.IMREAD_GRAYSCALE)     # グレースケール画像
    # clothes_mask_img_gs = None
    hist = cv2.calcHist([hsv], [0], clothes_mask_img_gs, [bins+1], [0, 180])
    # hist = cv2.normalize(hist, None, 0.0, 1.0, cv2.NORM_MINMAX)

    total_sum = sum(hist)
    for i in range(len(hist)):
        hist[i] = hist[i] / total_sum

    return hist

def comare_hist(img1,img2):
    METHOD = cv2.HISTCMP_CORREL

    q_hist = hue_hist(img1)
    t_hist = hue_hist(img2)
    print("len(t_hist) ", len(t_hist))
    plot_hist(q_hist)
    res = cv2.compareHist(q_hist, t_hist, METHOD)
    print("com hist {}".format(res))

def comare_hist_v1(q_hist, t_hist):
    METHOD = cv2.HISTCMP_CORREL
    print(len(q_hist), len(t_hist), type(q_hist), type(t_hist))
    res = cv2.compareHist(q_hist, t_hist, METHOD)
    plot_hist_do(q_hist, t_hist)
    print("com hist {}".format(res))
    # print(hist, type(hist))

import matplotlib.pyplot as plt
def plot_hist(hist, title=None):
    plt.plot(hist, color='blue', label='blue')                # 青チャンネルのヒストグラムのグラフを表示
    plt.legend(loc=0)                                                # 凡例
    plt.xlabel('Brightness')                                         # x軸ラベル(明度)
    plt.ylabel('Count')                                              # y軸ラベル(画素の数)
    plt.show()                                                       # ウィンドウにプロットを表示

def plot_hist_do(hist1, hist2, title=None):
    plt.plot(hist1, color='blue', label='blue')                # 青チャンネルのヒストグラムのグラフを表示
    plt.plot(hist2, color='red', label='teacher')                # 青チャンネルのヒストグラムのグラフを表示
    plt.legend(loc=0)                                                # 凡例
    plt.xlabel('Brightness')                                         # x軸ラベル(明度)
    plt.ylabel('Count')                                              # y軸ラベル(画素の数)
    plt.show()                                                       # ウィンドウにプロットを表示

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#    # h = hsv[:,:,0]

#     ax.hist(hist, bins=180)
#     ax.set_title('first histogram $\mu=100,\ \sigma=15$')
#     ax.set_xlabel('x')
#     ax.set_ylabel('freq')
#     plt.savefig('./hist_.jpg')
    # fig.show()

    # ax.set_xticks([0, 255])
    # ax.set_xlim([0, 255])
    # ax.set_xlabel("Pixel Value")
    # if title:
    #     ax.set_title(title)

    # bins = np.linspace(0, 255, 256)
    # ax.plot(bins, hist, color="k")

    # plt.show()


def main():
    feature_matching = FeatureMatching(method='akaze')

    if 1:
        from collections import defaultdict
        cloth_segmentation_dict = defaultdict(list)
        import glob
        for map_path in sorted(glob.glob('./refImg/*.jpg')):
            map_image = cv2.imread(map_path)
            map_clothes = cloth_segmentation(cv_image=map_image)
            keypoints_map, descriptors_map = feature_matching.extract_features(map_clothes)

            cloth_segmentation_dict['id'].append(map_path)
            cloth_segmentation_dict['clothes'].append(map_clothes)
            cloth_segmentation_dict['keypoints'].append(keypoints_map)
            cloth_segmentation_dict['descriptors'].append(descriptors_map)
            cloth_segmentation_dict['hue_hist'].append(hue_hist(map_clothes))       # TODO: hue_histの中にcloth_seg

    while True:
        cv_image = get_image()
        clothes = cloth_segmentation(cv_image=cv_image)
        p = 'cloth-seg.jpg'
        cv2.imwrite(p, clothes)

        # ターゲット画像の特徴量を抽出
        keypoints_target, descriptors_target = feature_matching.extract_features(clothes)

        # import glob
        # for map_path in sorted(glob.glob('./refImg/*.jpg')):

        for i in range(len(cloth_segmentation_dict['id'])):
            if 0:
                map_image = cv2.imread(map_path)
                map_clothes = cloth_segmentation(cv_image=map_image)
                keypoints_map, descriptors_map = feature_matching.extract_features(map_clothes)
            else:
                id = cloth_segmentation_dict['id'][i]
                map_clothes = cloth_segmentation_dict['clothes'][i]
                keypoints_map = cloth_segmentation_dict['keypoints'][i]
                descriptors_map = cloth_segmentation_dict['descriptors'][i]
                hue_hist_map = cloth_segmentation_dict['hue_hist'][i]
                
            
            # 特徴量マッチングを行い、探索画像を画像から検出
            result_image, flag = feature_matching.match_features(map_clothes, keypoints_map, descriptors_map, clothes, keypoints_target, descriptors_target)
            p = 'cloth-seg-after-matching.jpg'
            cv2.imwrite(p, result_image)
            print("{}, {}".format(id, flag))


            comare_hist_v1(hue_hist(clothes), hue_hist_map)
        
            def debug(image):
                img2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h, s, v = img2[:,:,0], img2[:,:,1], img2[:,:,2]
                hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
                hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
                hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
                plt.plot(hist_h, color='r', label="h")
                plt.plot(hist_s, color='g', label="s")
                plt.plot(hist_v, color='b', label="v")
                plt.legend()
                plt.show()
            # debug(clothes)



if __name__ == "__main__":
    main()