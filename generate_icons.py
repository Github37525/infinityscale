import os
from PIL import Image, ImageDraw

def create_logo(size):
    # 创建暗蓝色背景
    img = Image.new("RGBA", (size, size), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    
    # 绘制科技感圆角矩形
    padding = size * 0.1
    # 绘制现代发光渐变边缘
    draw.rounded_rectangle(
        [padding, padding, size - padding, size - padding],
        radius=size * 0.2,
        outline=(56, 189, 248, 255), # 亮蓝色 border
        width=max(2, int(size * 0.02))
    )
    
    # 在中心绘制矢量盒子标志（类似网页Logo）
    box_size = size * 0.25
    center = size / 2
    
    # 绘制四个小方块，带有现代半透明和发光效果
    spacing = size * 0.05
    # 左上
    draw.rectangle(
        [center - box_size - spacing, center - box_size - spacing, center - spacing, center - spacing],
        fill=(56, 189, 248, 255) # Sky blue
    )
    # 右上
    draw.rectangle(
        [center + spacing, center - box_size - spacing, center + box_size + spacing, center - spacing],
        fill=(14, 165, 233, 200)
    )
    # 左下
    draw.rectangle(
        [center - box_size - spacing, center + spacing, center - spacing, center + box_size + spacing],
        fill=(3, 105, 161, 200)
    )
    # 右下（点缀发光的绿色，代表高质量）
    draw.rectangle(
        [center + spacing, center + spacing, center + box_size + spacing, center + box_size + spacing],
        fill=(52, 211, 153, 255) # Emerald green
    )
    
    # 绘制连接网格线，凸显“矢量/印刷”主题
    draw.line([center - box_size, center, center + box_size, center], fill=(255, 255, 255, 80), width=max(1, int(size * 0.01)))
    draw.line([center, center - box_size, center, center + box_size], fill=(255, 255, 255, 80), width=max(1, int(size * 0.01)))
    
    return img

def main():
    print("Generating premium PWA icons...")
    os.makedirs("icons", exist_ok=True)
    
    icon_192 = create_logo(192)
    icon_192.save("icons/icon-192.png", "PNG")
    print("Created icons/icon-192.png")
    
    icon_512 = create_logo(512)
    icon_512.save("icons/icon-512.png", "PNG")
    print("Created icons/icon-512.png")
    
    print("PWA Icons generated successfully!")

if __name__ == "__main__":
    main()
