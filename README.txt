# RDN-tensorflow
the tf code of " Residual Dense Network for Image Super-Resolution"
The upsample scale is x2

Test:
Download model from '链接：https://pan.baidu.com/s/1QQ4kbUUolIQ8tRz25epvYw 密码：7aqo' and Put them in the 'model' folder.
Put your images in right input floder and set your output path in 'eval.py'
Run 'python eval.py'
The tensorboard output will be found in 'out/logs',and 'out/images' is the output images saved during training parse.Run'tensorboard --logdir=logs/ --port=8001' in 'model' folder.
