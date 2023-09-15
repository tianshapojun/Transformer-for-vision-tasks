import argparse
from .backbone import build_backbone

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    
    parser.add_argument('--backbone', default='resnet50', type=str,  # resnet50
                        help="Name of the convolutional backbone to use")
    # 使用正余弦位置编码还是可学习的位置编码
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # 其他参数的学习率
    parser.add_argument('--lr', default=1e-4, type=float)
    # backbone参数的学习率
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters  是否冻结bn层
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    # backbone是否使用空洞卷积
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # * Transformer
    # backbone输入transformer特征的维度
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    backbone = build_backbone(args)
    # out: list{0: tensor=[bs,2048,19,26] + mask=[bs,19,26]}  经过backbone resnet50 block5输出的结果
    # pos: list{0: [bs,256,19,26]}  位置编码
    features, pos = backbone(samples)
