"""OpenCV 兼容工具。"""


def video_writer_fourcc(code: str, cv2_module) -> int:
    """兼容不同 OpenCV 绑定暴露的 fourcc API。"""
    if hasattr(cv2_module, "VideoWriter_fourcc"):
        return cv2_module.VideoWriter_fourcc(*code)
    return cv2_module.VideoWriter.fourcc(*code)
