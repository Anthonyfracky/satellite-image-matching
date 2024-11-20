import cv2
import matplotlib.pyplot as plt
from image_processing import read_image, process_images, visualize_matches


def main():
    image1_path = 'images/NDdP_1.png'
    image2_path = 'images/NDdP_2.png'
    output_filename = 'Notre_Dame_matched_image.jpg'
    max_matches = 120

    print(f"Reading image {image1_path}")
    image1 = read_image(image1_path)
    print(f"Reading image {image2_path}\n")
    image2 = read_image(image2_path)

    results = process_images(image1_path, image2_path, max_matches=max_matches)

    global_kp1 = [kp for tile in results['keypoint_tiles1'] for kp in tile['keypoints']]
    global_kp2 = [kp for tile in results['keypoint_tiles2'] for kp in tile['keypoints']]

    matched_image = visualize_matches(image1, image2, global_kp1, global_kp2, results['matches'])

    cv2.imwrite(output_filename, matched_image)
    print(f"\nMatched image saved as {output_filename}")

    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title('Matched Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
