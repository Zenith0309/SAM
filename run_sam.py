import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load SAM model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

predictor = SamPredictor(sam)

# 2. Load image
image = cv2.imread("example.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)


# # 3. Use a point as prompt
# input_point = np.array([[500, 200]])  # 你可以改变这个点的坐标
# input_label = np.array([1])           # 1=前景点

# # 4. Predict mask
# masks, scores, _ = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )


# 3. Use bounding box as prompt
input_box = np.array([100, 50, 400, 350])  # [x1, y1, x2, y2]

# 4. Predict mask
masks, scores, _ = predictor.predict(
    box=input_box,
    multimask_output=False
)

# 5. Visualize
mask = masks[0]
output = image.copy()
output[mask] = [0, 255, 0]    # 绿色覆盖

cv2.imwrite("result.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

print("Saved to result.jpg")
