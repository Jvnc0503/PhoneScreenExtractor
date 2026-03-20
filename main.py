import argparse
import cv2
import numpy as np

def order_corners(corners: np.ndarray) -> np.ndarray:
    """Return corners ordered as top-left, top-right, bottom-right, bottom-left."""
    points = corners.reshape(4, 2).astype(np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(4)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def find_corners(image: np.ndarray) -> np.ndarray:
    """Find the 4 corners of the phone screen in the image."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    height, width = image.shape[0], image.shape[1]
    image_area = float(width * height)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), dtype=np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.05:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx)

    if not contours:
        raise ValueError("No contours were detected in the image.")

    # Fallback: approximate the largest contour as a rotated rectangle.
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    return order_corners(box)

def crop(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Crop image to the axis-aligned bounding box of the detected corners."""
    ordered = order_corners(corners)
    x_coords = ordered[:, 0]
    y_coords = ordered[:, 1]

    x_min = max(int(np.floor(np.min(x_coords))), 0)
    y_min = max(int(np.floor(np.min(y_coords))), 0)
    x_max = min(int(np.ceil(np.max(x_coords))), image.shape[1])
    y_max = min(int(np.ceil(np.max(y_coords))), image.shape[0])

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid crop bounds produced from corners.")

    return image[y_min:y_max, x_min:x_max]

def perspective_correction(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp image using corner points into a straight rectangle."""
    ordered = order_corners(corners)
    top_left, top_right, bottom_right, bottom_left = ordered

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    target_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    target_height = int(max(height_left, height_right))

    if target_width <= 0 or target_height <= 0:
        raise ValueError("Invalid target size for perspective correction.")

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (target_width, target_height))

def extract(image: np.ndarray) -> np.ndarray:
    """Extract phone screen as a straight rectangular image."""
    corners = find_corners(image)

    ordered = order_corners(corners)
    x_min = max(int(np.floor(np.min(ordered[:, 0]))), 0)
    y_min = max(int(np.floor(np.min(ordered[:, 1]))), 0)
    cropped = crop(image, ordered)

    # Shift corners into the cropped image coordinate system.
    shifted = ordered.copy()
    shifted[:, 0] -= x_min
    shifted[:, 1] -= y_min

    return perspective_correction(cropped, shifted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract phone screen from a photo.")
    parser.add_argument("--input", help="Input image path")
    parser.add_argument("--output", default="output/img.png", help="Output image path")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise ValueError(f"Could not read input image: {args.input}")

    result = extract(image)
    ok = cv2.imwrite(args.output, result)
    if not ok:
        raise ValueError(f"Could not write output image: {args.output}")


if __name__ == "__main__":
    main()