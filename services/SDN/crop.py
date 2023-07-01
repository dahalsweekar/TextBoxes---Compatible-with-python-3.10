import PIL
import matplotlib.pyplot as plt
from nms import nms

def image_cropper(image_path,image_name, dt_results):
    img = PIL.Image.open(image_path).convert('L')
    angle = 270
    img = img.rotate(angle, expand=True)
    img.save(f'./examples/crops/{image_name}')
    nms_flag = nms(dt_results, 0.3)
    i = 0
    for k, dt in enumerate(dt_results):
        if nms_flag[k]:
            name = '%.2f' % (dt[8])
            xmin = dt[0]
            ymin = dt[1]
            xmax = dt[2]
            ymax = dt[5]
            print (xmin,ymin,xmax,ymax)
            img = PIL.Image.open(image_path).convert('L')
            angle = 270
            img = img.rotate(angle, expand=True)
            img = img.crop((xmin,ymin,xmin+(xmax-xmin),ymin+(ymax-ymin)))
            img.save(f'./examples/crops/{i}{image_name}')
            i = i + 1