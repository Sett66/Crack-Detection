from PIL import Image


def resize_image_direct(input_path: str, output_path: str, target_size=(256, 256)):
    """
    直接将图片缩放到 256×256（不保持原比例，不填充，直接拉伸/压缩）
    """
    try:
        with Image.open(input_path) as img:
            # 直接调整到目标尺寸，使用高质量压缩算法
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            # 保存图片（JPG 格式指定质量，其他格式默认）
            resized_img.save(output_path, quality=95 if output_path.lower().endswith(('.jpg', '.jpeg')) else None)
            print(f"处理成功！输出路径：{output_path}")
    except Exception as e:
        print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    # 只需修改这两个路径
    INPUT_IMAGE = "/Crack_detection/LLH/Crack_datasets/DeepCrack_1/test_img/11231-8.jpg"  # 你的输入图片路径（如 "photo.png"、"D:/images/test.jpg"）
    OUTPUT_IMAGE = "/Crack_detection/LLH/11231-8.png"  # 输出图片路径（如 "result_256.jpg"）

    resize_image_direct(INPUT_IMAGE, OUTPUT_IMAGE)
