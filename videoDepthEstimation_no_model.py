import cv2
import numpy as np

def normalize_disparity_map(disparity):
    """Normalize disparity map for visualization"""
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disp_norm

def create_depth_map_visualization(disparity, min_depth=0, max_depth=255):
    """Convert disparity to a colored depth visualization"""
    # Normalize disparity to 0-255 range
    normalized = normalize_disparity_map(disparity)
    
    # Create a color map
    color_map = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    return color_map

# Initialize video
cap = cv2.VideoCapture("video.mp4")

# Create stereo matcher
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

cv2.namedWindow("Depth Comparison", cv2.WINDOW_NORMAL)    
while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:    
            break
    except:
        continue

    # Extract the left and right images
    left_img = frame[:,:frame.shape[1]//3]
    right_img = frame[:,frame.shape[1]//3:frame.shape[1]*2//3]
    color_real_depth = frame[:,frame.shape[1]*2//3:]

    # Convert to grayscale for stereo matching
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereo.compute(left_gray, right_gray)
    
    # Create colored visualization of the depth map
    color_depth = create_depth_map_visualization(disparity)
    
    # Resize to match original image size if needed
    color_depth = cv2.resize(color_depth, (left_img.shape[1], left_img.shape[0]))
    
    # Combine images for display
    combined_image = np.hstack((left_img, color_real_depth, color_depth))
    
    # Add FPS text (optional)
    combined_image = cv2.putText(combined_image, 'OpenCV Stereo', (50,50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Depth Comparison", combined_image)

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()