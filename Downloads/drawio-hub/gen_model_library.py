#!/usr/bin/env python3

import base64
import html
import json
import urllib.parse
import zlib


STYLES = {
    'input': 'shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#F8CECC;strokeColor=#B85450;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;verticalAlign=middle;',
    'conv': 'shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=13;fontFamily=Times New Roman;verticalAlign=middle;',
    'conv_sm': 'shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=8;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=11;fontFamily=Times New Roman;verticalAlign=middle;',
    'pool': 'shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#D5E8D4;strokeColor=#82B366;fontColor=#333333;strokeWidth=1.5;fontSize=13;fontFamily=Times New Roman;verticalAlign=middle;',
    'fc': 'shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=8;fillColor=#FFE6CC;strokeColor=#D79B00;fontColor=#333333;strokeWidth=1.5;fontSize=13;fontFamily=Times New Roman;verticalAlign=middle;',
    'output': 'shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=8;fillColor=#FFF2CC;strokeColor=#D6B656;fontColor=#333333;strokeWidth=1.5;fontSize=13;fontFamily=Times New Roman;verticalAlign=middle;',
    'deconv': 'shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=10;fillColor=#E1D5E7;strokeColor=#9673A6;fontColor=#333333;strokeWidth=1.5;fontSize=13;fontFamily=Times New Roman;verticalAlign=middle;',

    'relu': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'sigmoid': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'gelu': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'leakyrelu': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'tanh': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'softmax': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFF2CC;strokeColor=#D6B656;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'bn': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#999999;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'ln': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#999999;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'gn': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#999999;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'dropout': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#999999;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'flatten': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#999999;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'gap': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'maxpool': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'avgpool': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'embedding': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E1D5E7;strokeColor=#9673A6;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'mask': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#333333;strokeColor=#1A1A1A;fontColor=#FFFFFF;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',

    'mha': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'ffn': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'self_attn': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E1D5E7;strokeColor=#9673A6;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'cross_attn': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F8CECC;strokeColor=#B85450;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'pos_enc': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFF2CC;strokeColor=#D6B656;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'cls_token': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E1D5E7;strokeColor=#9673A6;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;fontStyle=1;',
    'patch': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=11;fontFamily=Times New Roman;',
    'seq_input': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',
    'heatmap': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F8CECC;strokeColor=#B85450;fontColor=#333333;strokeWidth=1.5;fontSize=12;fontFamily=Times New Roman;',

    'q': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;fontColor=#333333;strokeWidth=1.5;fontSize=13;fontFamily=Times New Roman;fontStyle=1;',
    'k': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=13;fontFamily=Times New Roman;fontStyle=1;',
    'v': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontColor=#333333;strokeWidth=1.5;fontSize=13;fontFamily=Times New Roman;fontStyle=1;',

    'add': 'shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=18;fontFamily=Times New Roman;fontStyle=1;',
    'multiply': 'shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=18;fontFamily=Times New Roman;fontStyle=1;',
    'concat': 'shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;fontStyle=1;',
    'matmul': 'shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;fontStyle=1;',
    'scale': 'shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=18;fontFamily=Times New Roman;fontStyle=1;',

    'container_gray': 'rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 4;fillColor=none;strokeColor=#999999;strokeWidth=2;verticalAlign=top;align=left;spacingLeft=8;spacingTop=5;fontColor=#999999;fontSize=14;fontStyle=1;fontFamily=Times New Roman;',
    'container_blue': 'rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 4;fillColor=none;strokeColor=#6C8EBF;strokeWidth=2;verticalAlign=top;align=left;spacingLeft=8;spacingTop=5;fontColor=#6C8EBF;fontSize=14;fontStyle=1;fontFamily=Times New Roman;',
    'container_green': 'rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 4;fillColor=none;strokeColor=#82B366;strokeWidth=2;verticalAlign=top;align=left;spacingLeft=8;spacingTop=5;fontColor=#82B366;fontSize=14;fontStyle=1;fontFamily=Times New Roman;',
    'container_red': 'rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 4;fillColor=none;strokeColor=#B85450;strokeWidth=2;verticalAlign=top;align=left;spacingLeft=8;spacingTop=5;fontColor=#B85450;fontSize=14;fontStyle=1;fontFamily=Times New Roman;',
    'container_purple': 'rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 4;fillColor=none;strokeColor=#9673A6;strokeWidth=2;verticalAlign=top;align=left;spacingLeft=8;spacingTop=5;fontColor=#9673A6;fontSize=14;fontStyle=1;fontFamily=Times New Roman;',
    'container_repeat': 'rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 4;fillColor=none;strokeColor=#666666;strokeWidth=2;verticalAlign=bottom;align=right;spacingRight=8;spacingBottom=5;fontColor=#666666;fontSize=13;fontStyle=1;fontFamily=Times New Roman;',

    'encoder_h': 'shape=trapezoid;whiteSpace=wrap;html=1;fixedSize=1;size=15;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;perimeter=trapezoidPerimeter;direction=east;',
    'decoder_h': 'shape=trapezoid;whiteSpace=wrap;html=1;fixedSize=1;size=15;fillColor=#D5E8D4;strokeColor=#82B366;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;perimeter=trapezoidPerimeter;direction=west;',
    'encoder_v': 'shape=trapezoid;whiteSpace=wrap;html=1;fixedSize=1;size=15;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;perimeter=trapezoidPerimeter;direction=south;',
    'decoder_v': 'shape=trapezoid;whiteSpace=wrap;html=1;fixedSize=1;size=15;fillColor=#D5E8D4;strokeColor=#82B366;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;perimeter=trapezoidPerimeter;',

    'lstm': 'rounded=1;arcSize=20;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;',
    'gru': 'rounded=1;arcSize=20;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontColor=#333333;strokeWidth=1.5;fontSize=14;fontFamily=Times New Roman;',

    'noise': 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#999999;fontColor=#333333;strokeWidth=1.5;fontSize=10;fontFamily=Times New Roman;dashed=1;dashPattern=4 4;',

    'branch': 'rhombus;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#666666;fontColor=#333333;strokeWidth=1.5;fontSize=10;fontFamily=Times New Roman;',

    'arrow': 'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#333333;strokeWidth=1.5;endSize=2;endArrow=block;endFill=1;fontSize=12;fontFamily=Times New Roman;fontColor=#333333;',
    'arrow_seg': 'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#333333;strokeWidth=1.5;endArrow=none;endFill=0;fontSize=12;fontFamily=Times New Roman;fontColor=#333333;',
    'arrow_skip': 'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#333333;strokeWidth=1.5;endSize=2;endArrow=block;endFill=1;fontSize=12;fontFamily=Times New Roman;fontColor=#333333;dashed=1;dashPattern=8 4;',
    'arrow_skip_seg': 'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#333333;strokeWidth=1.5;endArrow=none;endFill=0;fontSize=12;fontFamily=Times New Roman;fontColor=#333333;dashed=1;dashPattern=8 4;',
    'arrow_curved': 'curved=1;html=1;strokeColor=#333333;strokeWidth=1.5;endSize=2;endArrow=block;endFill=1;fontSize=12;fontFamily=Times New Roman;fontColor=#333333;',
    'arrow_bidir': 'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#333333;strokeWidth=1.5;startSize=2;startArrow=block;startFill=1;endSize=2;endArrow=block;endFill=1;fontSize=12;fontFamily=Times New Roman;fontColor=#333333;',
    'arrow_label': 'edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#333333;strokeWidth=1.5;endSize=2;endArrow=block;endFill=1;fontSize=11;fontFamily=Times New Roman;fontColor=#333333;',

    'dim_label': 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=10;fontColor=#999999;fontFamily=Times New Roman;',
    'caption': 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor=#333333;fontSize=14;fontFamily=Times New Roman;fontStyle=2;',

    'dot': 'shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#333333;strokeColor=#333333;strokeWidth=1.5;fontSize=0;fontFamily=Times New Roman;',
    'ellipsis': 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor=#999999;fontSize=16;fontStyle=1;fontFamily=Times New Roman;',
}


def fmt_num(value):
    if isinstance(value, int):
        return str(value)
    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return ('%.2f' % as_float).rstrip('0').rstrip('.')


def esc(value):
    return html.escape(str(value), quote=True)


def vertex(cell_id, value, x, y, w, h, style):
    return (
        f'<mxCell id="{esc(cell_id)}" value="{esc(value)}" style="{esc(style)}" '
        f'vertex="1" parent="g"><mxGeometry x="{fmt_num(x)}" y="{fmt_num(y)}" '
        f'width="{fmt_num(w)}" height="{fmt_num(h)}" as="geometry"/></mxCell>'
    )


def edge(cell_id, sx, sy, tx, ty, style, label=''):
    return (
        f'<mxCell id="{esc(cell_id)}" value="{esc(label)}" style="{esc(style)}" '
        f'edge="1" parent="g"><mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{fmt_num(sx)}" y="{fmt_num(sy)}" as="sourcePoint"/>'
        f'<mxPoint x="{fmt_num(tx)}" y="{fmt_num(ty)}" as="targetPoint"/>'
        f'</mxGeometry></mxCell>'
    )


def group_model(title, w, h, children_xml):
    del title
    return (
        '<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
        f'<mxCell id="g" value="" style="group;" vertex="1" connectable="0" parent="1">'
        f'<mxGeometry width="{fmt_num(w)}" height="{fmt_num(h)}" as="geometry"/></mxCell>'
        f'{children_xml}</root></mxGraphModel>'
    )


def encode_xml(xml_str):
    encoded = urllib.parse.quote(xml_str, safe='')
    compressed = zlib.compress(encoded.encode('utf-8'), 9)
    # Remove zlib header (first 2 bytes) and checksum (last 4 bytes) for raw deflate
    raw = compressed[2:-4]
    return base64.b64encode(raw).decode('utf-8')


def build_library(models):
    return '<mxlibrary>' + json.dumps(models, separators=(',', ':')) + '</mxlibrary>'


class Diagram:
    def __init__(self):
        self.children = []
        self.vertex_index = 1
        self.edge_index = 1
        self.max_x = 0.0
        self.max_y = 0.0

    def add_vertex(self, value, x, y, w, h, style_key):
        style = STYLES.get(style_key, style_key)
        cell_id = f'c{self.vertex_index}'
        self.vertex_index += 1
        self.children.append(vertex(cell_id, value, x, y, w, h, style))
        self.max_x = max(self.max_x, x + w)
        self.max_y = max(self.max_y, y + h)
        return {'id': cell_id, 'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h)}

    def add_edge(self, sx, sy, tx, ty, style_key='arrow', label=''):
        style = STYLES.get(style_key, style_key)
        cell_id = f'e{self.edge_index}'
        self.edge_index += 1
        self.children.append(edge(cell_id, sx, sy, tx, ty, style, label))
        self.max_x = max(self.max_x, sx, tx)
        self.max_y = max(self.max_y, sy, ty)
        return cell_id

    def size(self, margin=5):
        return int(self.max_x + margin), int(self.max_y + margin)


def cx(node):
    return node['x'] + node['w'] / 2


def cy(node):
    return node['y'] + node['h'] / 2


def top(node):
    return cx(node), node['y']


def bottom(node):
    return cx(node), node['y'] + node['h']


def left(node):
    return node['x'], cy(node)


def right(node):
    return node['x'] + node['w'], cy(node)


def connect_down(diagram, src, dst, style='arrow', label=''):
    sx, sy = bottom(src)
    tx, ty = top(dst)
    diagram.add_edge(sx, sy, tx, ty, style, label)


def connect_right(diagram, src, dst, style='arrow', label=''):
    sx, sy = right(src)
    tx, ty = left(dst)
    diagram.add_edge(sx, sy, tx, ty, style, label)


def connect_points(diagram, sx, sy, tx, ty, style='arrow', label=''):
    diagram.add_edge(sx, sy, tx, ty, style, label)


def segment_style(style):
    if style == 'arrow_skip':
        return 'arrow_skip_seg'
    return 'arrow_seg'


def connect_points_via_x(diagram, sx, sy, tx, ty, via_x, style='arrow'):
    seg_style = segment_style(style)
    connect_points(diagram, sx, sy, via_x, sy, seg_style)
    connect_points(diagram, via_x, sy, via_x, ty, seg_style)
    connect_points(diagram, via_x, ty, tx, ty, style)


def connect_points_via_y(diagram, sx, sy, tx, ty, via_y, style='arrow'):
    seg_style = segment_style(style)
    connect_points(diagram, sx, sy, sx, via_y, seg_style)
    connect_points(diagram, sx, via_y, tx, via_y, seg_style)
    connect_points(diagram, tx, via_y, tx, ty, style)


def connect_skip_vertical(diagram, src, dst, offset=25, target_side='right'):
    sx, sy = right(src)
    skip_x = max(sx, right(dst)[0]) + offset
    if target_side == 'left':
        tx, ty = left(dst)
    else:
        tx, ty = right(dst)
    connect_points_via_x(diagram, sx, sy, tx, ty, skip_x, 'arrow_skip')


def connect_skip_horizontal(diagram, src, dst, offset=20):
    sx, sy = top(src)
    tx, ty = top(dst)
    skip_y = min(sy, ty) - offset
    connect_points_via_y(diagram, sx, sy, tx, ty, skip_y, 'arrow_skip')


def make_entry(title, diagram):
    w, h = diagram.size()
    raw_xml = group_model(title, w, h, ''.join(diagram.children))
    return {'w': w, 'h': h, 'title': title, 'xml': encode_xml(raw_xml)}


def model_lenet5():
    d = Diagram()
    x = 70
    y = 10
    w = 95
    h = 42
    gap = 24
    n1 = d.add_vertex('Input<br>32x32x1', x, y, w, h, 'input')
    n2 = d.add_vertex('Conv<br>6@5x5', x, y + (h + gap) * 1, w, h, 'conv')
    n3 = d.add_vertex('Pool', x, y + (h + gap) * 2, w, h, 'pool')
    n4 = d.add_vertex('Conv<br>16@5x5', x, y + (h + gap) * 3, w, h, 'conv')
    n5 = d.add_vertex('Pool', x, y + (h + gap) * 4, w, h, 'pool')
    n6 = d.add_vertex('Flatten', x, y + (h + gap) * 5, w, h, 'flatten')
    n7 = d.add_vertex('FC 120', x, y + (h + gap) * 6, w, h, 'fc')
    n8 = d.add_vertex('FC 84', x, y + (h + gap) * 7, w, h, 'fc')
    n9 = d.add_vertex('Output 10', x, y + (h + gap) * 8, w, h, 'output')
    for src, dst in [(n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6), (n6, n7), (n7, n8), (n8, n9)]:
        connect_down(d, src, dst)
    return make_entry('LeNet-5', d)


def model_alexnet():
    d = Diagram()
    input_node = d.add_vertex('Input<br>224x224x3', 70, 10, 100, 40, 'input')
    d.add_vertex('Conv Stack', 38, 50, 164, 91, 'container_blue')
    conv_block = d.add_vertex('5 Conv + 3 Pool', 58, 85, 124, 36, 'conv_sm')
    flatten_node = d.add_vertex('Flatten', 70, 145, 100, 32, 'flatten')
    fc1 = d.add_vertex('FC1', 70, 201, 100, 34, 'fc')
    relu1 = d.add_vertex('ReLU', 82, 257, 76, 24, 'relu')
    drop1 = d.add_vertex('Dropout', 76, 303, 88, 24, 'dropout')
    fc2 = d.add_vertex('FC2', 70, 351, 100, 34, 'fc')
    relu2 = d.add_vertex('ReLU', 82, 407, 76, 24, 'relu')
    drop2 = d.add_vertex('Dropout', 76, 453, 88, 24, 'dropout')
    output_node = d.add_vertex('Output 1000', 70, 501, 100, 34, 'output')
    for src, dst in [
        (input_node, conv_block),
        (conv_block, flatten_node),
        (flatten_node, fc1),
        (fc1, relu1),
        (relu1, drop1),
        (drop1, fc2),
        (fc2, relu2),
        (relu2, drop2),
        (drop2, output_node),
    ]:
        connect_down(d, src, dst)
    return make_entry('AlexNet', d)


def model_vgg_block():
    d = Diagram()
    d.add_vertex('VGG Block', 40, 20, 140, 231, 'container_gray')
    conv1 = d.add_vertex('Conv 3x3', 60, 55, 100, 30, 'conv_sm')
    relu1 = d.add_vertex('ReLU', 76, 107, 68, 24, 'relu')
    conv2 = d.add_vertex('Conv 3x3', 60, 155, 100, 30, 'conv_sm')
    relu2 = d.add_vertex('ReLU', 76, 207, 68, 24, 'relu')
    for src, dst in [(conv1, relu1), (relu1, conv2), (conv2, relu2)]:
        connect_down(d, src, dst)
    return make_entry('VGG Block', d)


def model_resnet_block():
    d = Diagram()
    d.add_vertex('ResNet Basic Block', 50, 9, 150, 433, 'container_blue')
    inp = d.add_vertex('Input', 85, 44, 80, 28, 'seq_input')
    conv1 = d.add_vertex('Conv 3x3', 70, 96, 110, 30, 'conv_sm')
    bn1 = d.add_vertex('BN', 88, 150, 74, 24, 'bn')
    relu1 = d.add_vertex('ReLU', 88, 196, 74, 24, 'relu')
    conv2 = d.add_vertex('Conv 3x3', 70, 244, 110, 30, 'conv_sm')
    bn2 = d.add_vertex('BN', 88, 298, 74, 24, 'bn')
    add = d.add_vertex('+', 108, 342, 34, 34, 'add')
    out = d.add_vertex('ReLU', 88, 398, 74, 24, 'relu')
    for src, dst in [(inp, conv1), (conv1, bn1), (bn1, relu1), (relu1, conv2), (conv2, bn2), (bn2, add), (add, out)]:
        connect_down(d, src, dst)
    connect_skip_vertical(d, inp, add)
    return make_entry('ResNet Block', d)


def model_resnet_bottleneck():
    d = Diagram()
    d.add_vertex('ResNet Bottleneck', 52, 9, 156, 587, 'container_blue')
    inp = d.add_vertex('Input', 90, 44, 80, 28, 'seq_input')
    c1 = d.add_vertex('Conv 1x1', 72, 96, 116, 30, 'conv_sm')
    b1 = d.add_vertex('BN', 92, 150, 76, 24, 'bn')
    r1 = d.add_vertex('ReLU', 92, 196, 76, 24, 'relu')
    c2 = d.add_vertex('Conv 3x3', 72, 244, 116, 30, 'conv_sm')
    b2 = d.add_vertex('BN', 92, 298, 76, 24, 'bn')
    r2 = d.add_vertex('ReLU', 92, 344, 76, 24, 'relu')
    c3 = d.add_vertex('Conv 1x1', 72, 392, 116, 30, 'conv_sm')
    b3 = d.add_vertex('BN', 92, 446, 76, 24, 'bn')
    add = d.add_vertex('+', 113, 490, 34, 34, 'add')
    out = d.add_vertex('Output', 88, 548, 84, 28, 'output')
    for src, dst in [(inp, c1), (c1, b1), (b1, r1), (r1, c2), (c2, b2), (b2, r2), (r2, c3), (c3, b3), (b3, add), (add, out)]:
        connect_down(d, src, dst)
    connect_skip_vertical(d, inp, add)
    return make_entry('ResNet Bottleneck', d)


def model_densenet_block():
    d = Diagram()
    d.add_vertex('Dense Block', 64, 7, 142, 519, 'container_green')
    inp = d.add_vertex('Input', 95, 42, 80, 28, 'seq_input')
    bn1 = d.add_vertex('BN', 103, 94, 64, 22, 'bn')
    relu1 = d.add_vertex('ReLU', 99, 138, 72, 22, 'relu')
    c1 = d.add_vertex('Conv 1x1', 84, 184, 102, 28, 'conv_sm')
    d.add_vertex('...', 112, 236, 46, 24, 'ellipsis')
    bn2 = d.add_vertex('BN', 103, 282, 64, 22, 'bn')
    relu2 = d.add_vertex('ReLU', 99, 326, 72, 22, 'relu')
    c2 = d.add_vertex('Conv 3x3', 84, 372, 102, 28, 'conv_sm')
    concat_node = d.add_vertex('C', 117, 420, 34, 34, 'concat')
    out = d.add_vertex('Output', 90, 478, 88, 28, 'output')
    for src, dst in [(inp, bn1), (bn1, relu1), (relu1, c1), (c1, bn2), (bn2, relu2), (relu2, c2), (c2, concat_node), (concat_node, out)]:
        connect_down(d, src, dst)
    connect_skip_vertical(d, inp, concat_node)
    connect_skip_vertical(d, c1, concat_node)
    return make_entry('DenseNet Block', d)


def model_inception_module():
    d = Diagram()
    d.add_vertex('Inception Module', 8, 5, 321, 291, 'container_blue')
    inp = d.add_vertex('Input', 132, 40, 86, 28, 'seq_input')
    p1 = d.add_vertex('1x1', 28, 92, 56, 28, 'conv_sm')
    p2a = d.add_vertex('1x1', 103, 92, 56, 28, 'conv_sm')
    p2b = d.add_vertex('3x3', 103, 142, 56, 28, 'conv_sm')
    p3a = d.add_vertex('1x1', 178, 92, 56, 28, 'conv_sm')
    p3b = d.add_vertex('5x5', 178, 142, 56, 28, 'conv_sm')
    p4a = d.add_vertex('MaxPool', 253, 92, 56, 28, 'maxpool')
    p4b = d.add_vertex('1x1', 253, 142, 56, 28, 'conv_sm')
    concat_node = d.add_vertex('C', 158, 192, 34, 34, 'concat')
    out = d.add_vertex('Output', 130, 250, 90, 30, 'output')
    # Staggered fan-out from input to 4 branches
    inp_x, inp_y = bottom(inp)
    # Branch 1 (p1): via-point at y=78 (minimal offset)
    connect_points_via_y(d, inp_x, inp_y, cx(p1), p1['y'], 78)
    # Branch 2 (p2a): via-point at y=93 (+15pt stagger)
    connect_points_via_y(d, inp_x, inp_y, cx(p2a), p2a['y'], 93)
    # Branch 3 (p3a): via-point at y=108 (+30pt stagger)
    connect_points_via_y(d, inp_x, inp_y, cx(p3a), p3a['y'], 108)
    # Branch 4 (p4a): via-point at y=123 (+45pt stagger)
    connect_points_via_y(d, inp_x, inp_y, cx(p4a), p4a['y'], 123)
    connect_down(d, p2a, p2b)
    connect_down(d, p3a, p3b)
    connect_down(d, p4a, p4b)
    # Staggered fan-in to concat node
    # Branch 1 (p1): approach at concat_y - 20
    concat_y = concat_node['y']
    connect_points_via_y(d, cx(p1), p1['y'] + p1['h'], cx(concat_node), concat_y, concat_y - 20)
    # Branch 2 (p2b): approach at concat_y - 35 (+15pt stagger)
    connect_points_via_y(d, cx(p2b), p2b['y'] + p2b['h'], cx(concat_node), concat_y, concat_y - 35)
    # Branch 3 (p3b): approach at concat_y - 50 (+30pt stagger)
    connect_points_via_y(d, cx(p3b), p3b['y'] + p3b['h'], cx(concat_node), concat_y, concat_y - 50)
    # Branch 4 (p4b): approach at concat_y - 65 (+45pt stagger)
    connect_points_via_y(d, cx(p4b), p4b['y'] + p4b['h'], cx(concat_node), concat_y, concat_y - 65)
    connect_down(d, concat_node, out)
    return make_entry('Inception Module', d)


def model_mobilenet_block():
    d = Diagram()
    d.add_vertex('MobileNet Block', 52, 9, 156, 423, 'container_blue')
    inp = d.add_vertex('Input', 90, 44, 80, 28, 'seq_input')
    dw = d.add_vertex('DW Conv 3x3', 72, 96, 116, 30, 'conv_sm')
    bn1 = d.add_vertex('BN', 96, 150, 68, 22, 'bn')
    r1 = d.add_vertex('ReLU', 96, 194, 68, 22, 'relu')
    pw = d.add_vertex('PW Conv 1x1', 72, 240, 116, 30, 'conv_sm')
    bn2 = d.add_vertex('BN', 96, 294, 68, 22, 'bn')
    r2 = d.add_vertex('ReLU', 96, 338, 68, 22, 'relu')
    out = d.add_vertex('Output', 90, 384, 80, 28, 'output')
    for src, dst in [(inp, dw), (dw, bn1), (bn1, r1), (r1, pw), (pw, bn2), (bn2, r2), (r2, out)]:
        connect_down(d, src, dst)
    return make_entry('MobileNet Block', d)


def model_unet():
    d = Diagram()
    inp = d.add_vertex('Input', 35, 10, 70, 30, 'input')
    enc1 = d.add_vertex('Enc 1', 30, 64, 80, 34, 'conv')
    p1 = d.add_vertex('Pool', 45, 122, 50, 24, 'maxpool')
    enc2 = d.add_vertex('Enc 2', 30, 170, 80, 34, 'conv')
    p2 = d.add_vertex('Pool', 45, 228, 50, 24, 'maxpool')
    enc3 = d.add_vertex('Enc 3', 30, 276, 80, 34, 'conv')
    p3 = d.add_vertex('Pool', 45, 334, 50, 24, 'maxpool')
    bottleneck = d.add_vertex('Bottleneck', 150, 382, 95, 36, 'conv')
    up3 = d.add_vertex('Up', 275, 334, 65, 30, 'deconv')
    cat3 = d.add_vertex('C', 370, 336, 26, 26, 'concat')
    dec3 = d.add_vertex('Dec 3', 426, 332, 80, 34, 'deconv')
    up2 = d.add_vertex('Up', 275, 248, 65, 30, 'deconv')
    cat2 = d.add_vertex('C', 370, 250, 26, 26, 'concat')
    dec2 = d.add_vertex('Dec 2', 426, 246, 80, 34, 'deconv')
    up1 = d.add_vertex('Up', 275, 162, 65, 30, 'deconv')
    cat1 = d.add_vertex('C', 370, 164, 26, 26, 'concat')
    dec1 = d.add_vertex('Dec 1', 426, 160, 80, 34, 'deconv')
    out = d.add_vertex('Output', 426, 88, 80, 34, 'output')

    for src, dst in [(inp, enc1), (enc1, p1), (p1, enc2), (enc2, p2), (p2, enc3), (enc3, p3)]:
        connect_down(d, src, dst)
    connect_points_via_x(d, right(p3)[0], right(p3)[1], left(bottleneck)[0], left(bottleneck)[1], right(p3)[0] + 25)
    connect_right(d, bottleneck, up3)
    connect_right(d, up3, cat3)
    connect_right(d, cat3, dec3)
    connect_points_via_y(d, left(dec3)[0], left(dec3)[1], right(up2)[0], right(up2)[1], up2['y'] + up2['h'] + 12)
    connect_right(d, up2, cat2)
    connect_right(d, cat2, dec2)
    connect_points_via_y(d, left(dec2)[0], left(dec2)[1], right(up1)[0], right(up1)[1], up1['y'] + up1['h'] + 12)
    connect_right(d, up1, cat1)
    connect_right(d, cat1, dec1)
    connect_points(d, cx(dec1), dec1['y'], cx(out), out['y'] + out['h'])

    skip_bus_x = dec3['x'] + dec3['w'] + 30

    enc1_rx, enc1_ry = right(enc1)
    cat1_tx, cat1_ty = cx(cat1), cat1['y']
    connect_points(d, enc1_rx, enc1_ry, skip_bus_x, enc1_ry, 'arrow_skip_seg')
    connect_points(d, skip_bus_x, enc1_ry, skip_bus_x, cat1_ty - 10, 'arrow_skip_seg')
    connect_points(d, skip_bus_x, cat1_ty - 10, cat1_tx, cat1_ty - 10, 'arrow_skip_seg')
    connect_points(d, cat1_tx, cat1_ty - 10, cat1_tx, cat1_ty, 'arrow_skip')

    skip_bus_x2 = skip_bus_x - 15
    enc2_rx, enc2_ry = right(enc2)
    cat2_tx, cat2_ty = cx(cat2), cat2['y']
    connect_points(d, enc2_rx, enc2_ry, skip_bus_x2, enc2_ry, 'arrow_skip_seg')
    connect_points(d, skip_bus_x2, enc2_ry, skip_bus_x2, cat2_ty - 10, 'arrow_skip_seg')
    connect_points(d, skip_bus_x2, cat2_ty - 10, cat2_tx, cat2_ty - 10, 'arrow_skip_seg')
    connect_points(d, cat2_tx, cat2_ty - 10, cat2_tx, cat2_ty, 'arrow_skip')

    skip_bus_x3 = skip_bus_x - 30
    enc3_rx, enc3_ry = right(enc3)
    cat3_tx, cat3_ty = cx(cat3), cat3['y']
    connect_points(d, enc3_rx, enc3_ry, skip_bus_x3, enc3_ry, 'arrow_skip_seg')
    connect_points(d, skip_bus_x3, enc3_ry, skip_bus_x3, cat3_ty - 10, 'arrow_skip_seg')
    connect_points(d, skip_bus_x3, cat3_ty - 10, cat3_tx, cat3_ty - 10, 'arrow_skip_seg')
    connect_points(d, cat3_tx, cat3_ty - 10, cat3_tx, cat3_ty, 'arrow_skip')
    return make_entry('U-Net', d)


def model_fpn():
    d = Diagram()
    c5 = d.add_vertex('C5', 20, 40, 70, 32, 'conv')
    c4 = d.add_vertex('C4', 20, 110, 70, 32, 'conv')
    c3 = d.add_vertex('C3', 20, 180, 70, 32, 'conv')
    c2 = d.add_vertex('C2', 20, 250, 70, 32, 'conv')
    l5 = d.add_vertex('1x1', 120, 42, 55, 28, 'conv_sm')
    l4 = d.add_vertex('1x1', 120, 112, 55, 28, 'conv_sm')
    l3 = d.add_vertex('1x1', 120, 182, 55, 28, 'conv_sm')
    l2 = d.add_vertex('1x1', 120, 252, 55, 28, 'conv_sm')
    p5 = d.add_vertex('P5', 276, 42, 55, 28, 'conv_sm')
    add4 = d.add_vertex('+', 220, 113, 26, 26, 'add')
    p4 = d.add_vertex('P4', 276, 112, 55, 28, 'conv_sm')
    add3 = d.add_vertex('+', 220, 183, 26, 26, 'add')
    p3 = d.add_vertex('P3', 276, 182, 55, 28, 'conv_sm')
    add2 = d.add_vertex('+', 220, 253, 26, 26, 'add')
    p2 = d.add_vertex('P2', 276, 252, 55, 28, 'conv_sm')
    o5 = d.add_vertex('Out P5', 361, 42, 72, 28, 'output')
    o4 = d.add_vertex('Out P4', 361, 112, 72, 28, 'output')
    o3 = d.add_vertex('Out P3', 361, 182, 72, 28, 'output')
    o2 = d.add_vertex('Out P2', 361, 252, 72, 28, 'output')

    for src, dst in [(c5, l5), (c4, l4), (c3, l3), (c2, l2), (l5, p5), (p5, o5), (p4, o4), (p3, o3), (p2, o2)]:
        connect_right(d, src, dst)
    connect_points(d, cx(p5), p5['y'] + p5['h'], cx(add4), add4['y'])
    connect_right(d, l4, add4)
    connect_right(d, add4, p4)
    connect_points(d, cx(p4), p4['y'] + p4['h'], cx(add3), add3['y'])
    connect_right(d, l3, add3)
    connect_right(d, add3, p3)
    connect_points(d, cx(p3), p3['y'] + p3['h'], cx(add2), add2['y'])
    connect_right(d, l2, add2)
    connect_right(d, add2, p2)
    return make_entry('FPN', d)


def model_yolo_head():
    d = Diagram()
    rows = [40, 140, 240]
    for idx, y in enumerate(rows, start=1):
        feat = d.add_vertex(f'Feature S{idx}', 20, y, 90, 32, 'conv')
        pred = d.add_vertex('Pred Conv', 130, y, 90, 32, 'conv_sm')
        bbox = d.add_vertex('bbox', 250, y - 10, 58, 24, 'output')
        obj = d.add_vertex('obj', 338, y - 10, 58, 24, 'output')
        cls = d.add_vertex('class', 426, y - 10, 58, 24, 'output')
        connect_right(d, feat, pred)
        # Fan-out from pred to 3 outputs using staggered via-points
        pred_r_x, pred_r_y = right(pred)
        bbox_l_x, bbox_l_y = left(bbox)
        obj_l_x, obj_l_y = left(obj)
        cls_l_x, cls_l_y = left(cls)
        
        # Route 1 (bbox): direct route via pred_r_y
        connect_points_via_y(d, pred_r_x, pred_r_y, bbox_l_x, bbox_l_y, pred_r_y)
        # Route 2 (obj): staggered +15pt
        connect_points_via_y(d, pred_r_x, pred_r_y, obj_l_x, obj_l_y, pred_r_y + 15)
        # Route 3 (cls): staggered +30pt
        connect_points_via_y(d, pred_r_x, pred_r_y, cls_l_x, cls_l_y, pred_r_y + 30)
    return make_entry('YOLO Head', d)


def model_faster_rcnn():
    d = Diagram()
    backbone = d.add_vertex('Backbone', 20, 120, 82, 40, 'conv')
    d.add_vertex('RPN', 112, 50, 162, 125, 'container_red')
    rpn_conv = d.add_vertex('Conv', 150, 85, 86, 26, 'conv_sm')
    rpn_cls = d.add_vertex('RPN cls', 132, 133, 50, 22, 'output')
    rpn_reg = d.add_vertex('RPN reg', 204, 133, 50, 22, 'output')
    roi = d.add_vertex('RoI Pool', 304, 123, 84, 34, 'pool')
    fc_head = d.add_vertex('FC Head', 418, 123, 84, 34, 'fc')
    cls = d.add_vertex('Class', 532, 77, 84, 26, 'output')
    reg = d.add_vertex('BBox', 532, 169, 84, 26, 'output')

    connect_right(d, backbone, rpn_conv)
    connect_down(d, rpn_conv, rpn_cls)
    connect_down(d, rpn_conv, rpn_reg)
    rpn_container_bottom = 175
    rpn_container_right = 274
    bk_rx, bk_ry = right(backbone)
    roi_lx, roi_ly = left(roi)
    via_x = rpn_container_right + 10
    via_y = rpn_container_bottom + 15

    connect_points(d, bk_rx, bk_ry, via_x, bk_ry, 'arrow_seg')
    connect_points(d, via_x, bk_ry, via_x, via_y, 'arrow_seg')
    connect_points(d, via_x, via_y, roi_lx, via_y, 'arrow_seg')
    connect_points(d, roi_lx, via_y, roi_lx, roi_ly, 'arrow')
    connect_right(d, rpn_cls, roi, 'arrow_skip')
    connect_right(d, rpn_reg, roi, 'arrow_skip')
    connect_right(d, roi, fc_head)
    connect_points(d, right(fc_head)[0], right(fc_head)[1], left(cls)[0], left(cls)[1])
    connect_points(d, right(fc_head)[0], right(fc_head)[1], left(reg)[0], left(reg)[1])
    return make_entry('Faster R-CNN', d)


def model_fcn_decoder():
    d = Diagram()
    f5 = d.add_vertex('F5', 20, 30, 62, 28, 'conv')
    f4 = d.add_vertex('F4', 20, 100, 62, 28, 'conv')
    f3 = d.add_vertex('F3', 20, 170, 62, 28, 'conv')
    up1 = d.add_vertex('Up', 112, 30, 62, 28, 'deconv')
    add1 = d.add_vertex('+', 204, 96, 26, 26, 'add')
    up2 = d.add_vertex('Up', 260, 96, 62, 28, 'deconv')
    add2 = d.add_vertex('+', 352, 166, 26, 26, 'add')
    up3 = d.add_vertex('Up', 408, 166, 62, 28, 'deconv')
    heatmap = d.add_vertex('Heatmap', 500, 165, 70, 30, 'heatmap')

    connect_right(d, f5, up1)

    # Add1: Routes from up1 (top) and f4 (bottom) with staggered via-points

    # Up1 -> Add1: Route above add1 (via_y = 81)
    up1_rx, up1_ry = right(up1)
    add1_lx, add1_ly = left(add1)
    via_y_up1 = 81  # Above add1 (center at 96)
    connect_points(d, up1_rx, up1_ry, up1_rx + 15, up1_ry, 'arrow_seg')
    connect_points(d, up1_rx + 15, up1_ry, up1_rx + 15, via_y_up1, 'arrow_seg')
    connect_points(d, up1_rx + 15, via_y_up1, add1_lx, via_y_up1, 'arrow_seg')
    connect_points(d, add1_lx, via_y_up1, add1_lx, add1_ly, 'arrow')

    # F4 -> Add1: Route below add1 (via_y = 111)
    f4_rx, f4_ry = right(f4)
    via_y_f4 = 111  # Below add1 (center at 96)
    connect_points(d, f4_rx, f4_ry, f4_rx + 15, f4_ry, 'arrow_seg')
    connect_points(d, f4_rx + 15, f4_ry, f4_rx + 15, via_y_f4, 'arrow_seg')
    connect_points(d, f4_rx + 15, via_y_f4, add1_lx, via_y_f4, 'arrow_seg')
    connect_points(d, add1_lx, via_y_f4, add1_lx, add1_ly, 'arrow')

    connect_right(d, add1, up2)

    # Add2: Routes from up2 (top) and f3 (bottom) with staggered via-points

    # Up2 -> Add2: Route above add2 (via_y = 151)
    up2_rx, up2_ry = right(up2)
    add2_lx, add2_ly = left(add2)
    via_y_up2 = 151  # Above add2 (center at 166)
    connect_points(d, up2_rx, up2_ry, up2_rx + 15, up2_ry, 'arrow_seg')
    connect_points(d, up2_rx + 15, up2_ry, up2_rx + 15, via_y_up2, 'arrow_seg')
    connect_points(d, up2_rx + 15, via_y_up2, add2_lx, via_y_up2, 'arrow_seg')
    connect_points(d, add2_lx, via_y_up2, add2_lx, add2_ly, 'arrow')

    # F3 -> Add2: Route below add2 (via_y = 181)
    f3_rx, f3_ry = right(f3)
    via_y_f3 = 181  # Below add2 (center at 166)
    connect_points(d, f3_rx, f3_ry, f3_rx + 15, f3_ry, 'arrow_seg')
    connect_points(d, f3_rx + 15, f3_ry, f3_rx + 15, via_y_f3, 'arrow_seg')
    connect_points(d, f3_rx + 15, via_y_f3, add2_lx, via_y_f3, 'arrow_seg')
    connect_points(d, add2_lx, via_y_f3, add2_lx, add2_ly, 'arrow')

    connect_right(d, add2, up3)
    connect_right(d, up3, heatmap)
    return make_entry('FCN Decoder', d)


def model_aspp_module():
    d = Diagram()
    inp = d.add_vertex('Input', 20, 108, 72, 30, 'conv')
    d.add_vertex('ASPP', 108, 17, 292, 279, 'container_blue')
    b1 = d.add_vertex('Dilated 1', 128, 52, 80, 28, 'conv_sm')
    b2 = d.add_vertex('Dilated 6', 128, 102, 80, 28, 'conv_sm')
    b3 = d.add_vertex('Dilated 12', 128, 152, 80, 28, 'conv_sm')
    b4 = d.add_vertex('Dilated 18', 128, 202, 80, 28, 'conv_sm')
    gap = d.add_vertex('GAP', 238, 202, 70, 26, 'gap')
    proj = d.add_vertex('1x1', 238, 250, 70, 26, 'conv_sm')
    concat_node = d.add_vertex('C', 350, 138, 30, 30, 'concat')
    out = d.add_vertex('Conv 1x1', 410, 138, 86, 30, 'conv')

    src_x, src_y = right(inp)
    branch_bus_x = b1['x'] - 18
    for branch in [b1, b2, b3, b4, gap]:
        bx, by = left(branch)
        connect_points_via_x(d, src_x, src_y, bx, by, branch_bus_x)
    connect_down(d, gap, proj)
    merge_bus_x = concat_node['x'] - 20
    for branch in [b1, b2, b3, b4, proj]:
        sx, sy = right(branch)
        tx, ty = left(concat_node)
        connect_points_via_x(d, sx, sy, tx, ty, merge_bus_x)
    connect_right(d, concat_node, out)
    return make_entry('ASPP Module', d)


def model_transformer_encoder():
    d = Diagram()
    inp = d.add_vertex('Input', 110, 8, 80, 30, 'seq_input')
    d.add_vertex('Encoder Block', 75, 39, 150, 351, 'container_purple')
    mha = d.add_vertex('Multi-Head Attn', 95, 74, 110, 34, 'mha')
    add1 = d.add_vertex('+', 135, 128, 30, 30, 'add')
    ln1 = d.add_vertex('LN', 110, 184, 80, 26, 'ln')
    ffn = d.add_vertex('Feed Forward', 95, 234, 110, 34, 'ffn')
    add2 = d.add_vertex('+', 135, 288, 30, 30, 'add')
    ln2 = d.add_vertex('LN', 110, 344, 80, 26, 'ln')
    out = d.add_vertex('Output', 110, 394, 80, 30, 'output')

    for src, dst in [(inp, mha), (mha, add1), (add1, ln1), (ln1, ffn), (ffn, add2), (add2, ln2), (ln2, out)]:
        connect_down(d, src, dst)
    connect_skip_vertical(d, inp, add1)
    connect_skip_vertical(d, ln1, add2)
    return make_entry('Transformer Encoder', d)


def model_transformer_decoder():
    d = Diagram()
    inp = d.add_vertex('Input', 146, 8, 90, 30, 'seq_input')
    memory = d.add_vertex('Encoder Memory', 10, 228, 92, 28, 'embedding')
    d.add_vertex('Decoder Block', 116, 39, 164, 493, 'container_purple')
    msa = d.add_vertex('Masked Self-Attn', 136, 74, 124, 32, 'self_attn')
    add1 = d.add_vertex('+', 183, 126, 30, 30, 'add')
    ln1 = d.add_vertex('LN', 158, 178, 80, 26, 'ln')
    cross = d.add_vertex('Cross-Attn', 136, 228, 124, 32, 'cross_attn')
    add2 = d.add_vertex('+', 183, 280, 30, 30, 'add')
    ln2 = d.add_vertex('LN', 158, 332, 80, 26, 'ln')
    ffn = d.add_vertex('Feed Forward', 136, 382, 124, 32, 'ffn')
    add3 = d.add_vertex('+', 183, 434, 30, 30, 'add')
    ln3 = d.add_vertex('LN', 158, 486, 80, 26, 'ln')
    out = d.add_vertex('Output', 146, 538, 90, 30, 'output')

    for src, dst in [(inp, msa), (msa, add1), (add1, ln1), (ln1, cross), (cross, add2), (add2, ln2), (ln2, ffn), (ffn, add3), (add3, ln3), (ln3, out)]:
        connect_down(d, src, dst)
    connect_right(d, memory, cross)
    connect_skip_vertical(d, inp, add1)
    connect_skip_vertical(d, ln1, add2)
    connect_skip_vertical(d, ln2, add3)
    return make_entry('Transformer Decoder', d)


def model_full_transformer():
    d = Diagram()
    src_in = d.add_vertex('Src Input', 15, 92, 70, 30, 'seq_input')
    src_emb = d.add_vertex('Embedding', 115, 92, 80, 30, 'embedding')
    src_pos = d.add_vertex('Pos Enc', 115, 38, 80, 24, 'pos_enc')
    src_add = d.add_vertex('+', 225, 95, 24, 24, 'add')
    d.add_vertex('Encoder xN', 259, 57, 108, 98, 'container_blue')
    enc_blk = d.add_vertex('Encoder', 279, 92, 68, 30, 'mha')

    tgt_in = d.add_vertex('Tgt Input', 15, 234, 70, 30, 'seq_input')
    tgt_emb = d.add_vertex('Embedding', 115, 234, 80, 30, 'embedding')
    tgt_pos = d.add_vertex('Pos Enc', 115, 180, 80, 24, 'pos_enc')
    tgt_add = d.add_vertex('+', 225, 237, 24, 24, 'add')
    d.add_vertex('Decoder xN', 359, 199, 124, 116, 'container_green')
    dec_blk = d.add_vertex('Decoder', 379, 234, 74, 30, 'self_attn')

    linear = d.add_vertex('Linear', 483, 234, 70, 30, 'fc')
    out = d.add_vertex('Softmax', 583, 234, 55, 30, 'output')

    connect_right(d, src_in, src_emb)
    connect_right(d, src_emb, src_add)
    connect_points(d, cx(src_pos), src_pos['y'] + src_pos['h'], cx(src_add), src_add['y'])
    connect_right(d, src_add, enc_blk)

    connect_right(d, tgt_in, tgt_emb)
    connect_right(d, tgt_emb, tgt_add)
    connect_points(d, cx(tgt_pos), tgt_pos['y'] + tgt_pos['h'], cx(tgt_add), tgt_add['y'])
    connect_right(d, tgt_add, dec_blk)

    connect_points(d, right(enc_blk)[0], right(enc_blk)[1], left(dec_blk)[0], left(dec_blk)[1], 'arrow_label', 'cross')
    connect_right(d, dec_blk, linear)
    connect_right(d, linear, out)
    return make_entry('Full Transformer', d)


def model_vit():
    d = Diagram()
    patch = d.add_vertex('Patch Part.', 10, 88, 70, 30, 'patch')
    embed = d.add_vertex('Embedding', 110, 88, 70, 30, 'embedding')
    cls = d.add_vertex('[CLS]', 135, 42, 55, 24, 'cls_token')
    pos = d.add_vertex('Pos Enc', 135, 138, 55, 24, 'pos_enc')
    merge = d.add_vertex('+', 210, 91, 24, 24, 'add')
    d.add_vertex('Repeat xN', 254, 55, 115, 98, 'container_repeat')
    enc = d.add_vertex('Encoder', 264, 88, 65, 30, 'mha')
    mlp = d.add_vertex('MLP Head', 399, 88, 55, 30, 'fc')
    out = d.add_vertex('Output', 484, 88, 50, 30, 'output')

    connect_right(d, patch, embed)
    connect_right(d, embed, merge)
    # Fan-in from cls and pos to merge using staggered via-points
    cls_r_x, cls_r_y = right(cls)
    pos_r_x, pos_r_y = right(pos)
    merge_l_x, merge_l_y = left(merge)
    
    # Route 1 (cls → merge): via_x closer to merge
    connect_points_via_x(d, cls_r_x, cls_r_y, merge_l_x, merge_l_y, merge_l_x - 15)
    # Route 2 (pos → merge): via_x staggered -30pt
    connect_points_via_x(d, pos_r_x, pos_r_y, merge_l_x, merge_l_y, merge_l_x - 30)
    connect_right(d, merge, enc)
    connect_right(d, enc, mlp)
    connect_right(d, mlp, out)
    return make_entry('ViT', d)


def model_bert_block():
    d = Diagram()
    emb = d.add_vertex('Token+Seg+Pos Emb', 20, 100, 130, 34, 'embedding')
    d.add_vertex('Transformer xN', 180, 73, 130, 89, 'container_repeat')
    enc = d.add_vertex('Encoder', 200, 108, 90, 34, 'mha')
    cls = d.add_vertex('[CLS] Output', 320, 68, 90, 30, 'output')
    tok = d.add_vertex('Token Outputs', 320, 148, 90, 30, 'output')
    connect_right(d, emb, enc)
    split_x = right(enc)[0] + 24
    sx, sy = right(enc)
    cls_x, cls_y = left(cls)
    tok_x, tok_y = left(tok)
    connect_points(d, sx, sy, split_x, sy, 'arrow_seg')
    connect_points(d, split_x, sy, split_x, cls_y, 'arrow_seg')
    connect_points(d, split_x, cls_y, cls_x, cls_y)
    connect_points(d, split_x, sy, split_x, tok_y, 'arrow_seg')
    connect_points(d, split_x, tok_y, tok_x, tok_y)
    return make_entry('BERT Block', d)


def model_gpt_block():
    d = Diagram()
    inp = d.add_vertex('Input', 130, 10, 80, 30, 'seq_input')
    d.add_vertex('GPT Block', 50, 40, 230, 245, 'container_purple')
    mask = d.add_vertex('Mask', 15, 91, 60, 24, 'mask')
    msa = d.add_vertex('Masked SA + LN', 105, 75, 120, 32, 'self_attn')
    add1 = d.add_vertex('+', 195, 127, 30, 30, 'add')
    ffn = d.add_vertex('FFN + LN', 105, 181, 120, 32, 'ffn')
    add2 = d.add_vertex('+', 195, 235, 30, 30, 'add')
    out = d.add_vertex('Output', 130, 289, 80, 30, 'output')
    connect_down(d, inp, msa)
    connect_down(d, msa, add1)
    connect_down(d, add1, ffn)
    connect_down(d, ffn, add2)
    connect_down(d, add2, out)
    connect_right(d, mask, msa)
    connect_skip_vertical(d, inp, add1)
    connect_skip_vertical(d, add1, add2)
    return make_entry('GPT Block', d)


def model_multi_head_attention_detail():
    d = Diagram()
    inp = d.add_vertex('Input', 15, 120, 55, 28, 'seq_input')
    q = d.add_vertex('Q', 100, 40, 45, 26, 'q')
    k = d.add_vertex('K', 100, 120, 45, 26, 'k')
    v = d.add_vertex('V', 100, 200, 45, 26, 'v')
    mat1 = d.add_vertex('x', 180, 95, 32, 32, 'matmul')
    scale = d.add_vertex('/', 242, 95, 32, 32, 'scale')
    softmax = d.add_vertex('Softmax', 304, 96, 55, 30, 'softmax')
    mat2 = d.add_vertex('x', 389, 140, 32, 32, 'matmul')
    concat_node = d.add_vertex('C', 451, 140, 32, 32, 'concat')
    fc = d.add_vertex('FC Out', 513, 139, 70, 34, 'fc')

    # Staggered fan-out from input to Q, K, V using different via-points
    # Route 1 (to Q): Via-point at x=78
    connect_points_via_x(d, right(inp)[0], right(inp)[1], left(q)[0], left(q)[1], 78)
    # Route 2 (to K): Via-point at x=93 (middle, +15pt stagger)
    connect_points_via_x(d, right(inp)[0], right(inp)[1], left(k)[0], left(k)[1], 93)
    # Route 3 (to V): Via-point at x=108 (+30pt stagger)
    connect_points_via_x(d, right(inp)[0], right(inp)[1], left(v)[0], left(v)[1], 108)
    connect_points(d, right(q)[0], right(q)[1], left(mat1)[0], left(mat1)[1])
    connect_points(d, right(k)[0], right(k)[1], left(mat1)[0], left(mat1)[1])
    connect_right(d, mat1, scale)
    connect_right(d, scale, softmax)
    connect_points(d, right(softmax)[0], right(softmax)[1], left(mat2)[0], left(mat2)[1])
    connect_points(d, right(v)[0], right(v)[1], left(mat2)[0], left(mat2)[1])
    connect_right(d, mat2, concat_node)
    connect_right(d, concat_node, fc)
    return make_entry('Multi-Head Attn Detail', d)


def model_autoencoder():
    d = Diagram()
    inp = d.add_vertex('Input', 20, 70, 60, 34, 'input')
    enc = d.add_vertex('Encoder', 110, 65, 80, 44, 'encoder_h')
    latent = d.add_vertex('Latent z', 220, 74, 60, 28, 'fc')
    dec = d.add_vertex('Decoder', 310, 65, 80, 44, 'decoder_h')
    out = d.add_vertex('Output', 420, 70, 60, 34, 'output')
    for src, dst in [(inp, enc), (enc, latent), (latent, dec), (dec, out)]:
        connect_right(d, src, dst)
    return make_entry('Autoencoder', d)


def model_vae():
    d = Diagram()
    inp = d.add_vertex('Input', 10, 120, 60, 32, 'input')
    enc = d.add_vertex('Encoder', 100, 112, 80, 44, 'encoder_h')
    mu = d.add_vertex('mu', 210, 70, 56, 26, 'fc')
    sigma = d.add_vertex('sigma', 210, 174, 56, 26, 'fc')
    eps = d.add_vertex('eps', 300, 218, 50, 24, 'noise')
    mul = d.add_vertex('*', 380, 174, 26, 26, 'multiply')
    add = d.add_vertex('+', 436, 122, 26, 26, 'add')
    dec = d.add_vertex('Decoder', 492, 113, 80, 44, 'decoder_h')
    out = d.add_vertex('Output', 602, 119, 55, 32, 'output')
    d.add_vertex('Loss: KL + Recon', 200, 8, 240, 42, 'container_red')
    kl = d.add_vertex('KL', 220, 22, 70, 22, 'relu')
    recon = d.add_vertex('Recon', 320, 22, 82, 22, 'relu')

    connect_right(d, inp, enc)
    connect_points(d, right(enc)[0], right(enc)[1], left(mu)[0], left(mu)[1])
    connect_points(d, right(enc)[0], right(enc)[1], left(sigma)[0], left(sigma)[1])
    connect_right(d, sigma, mul)
    connect_points(d, right(eps)[0], right(eps)[1], left(mul)[0], left(mul)[1])
    connect_right(d, mu, add)
    connect_points(d, right(mul)[0], right(mul)[1], left(add)[0], left(add)[1])
    connect_right(d, add, dec)
    connect_right(d, dec, out)
    # Route loss connections via TOP bus (V-H-V pattern) to avoid crossing main flow
    top_bus_y = 40  # Above mu (y=70, -30 margin)
    
    # mu → KL: vertical up to top_bus, horizontal left, vertical down
    connect_points(d, cx(mu), mu['y'], cx(mu), top_bus_y, 'arrow_skip')  # V
    connect_points(d, cx(mu), top_bus_y, cx(kl), top_bus_y, 'arrow_skip')  # H
    connect_points(d, cx(kl), top_bus_y, cx(kl), kl['y'] + kl['h'], 'arrow_skip')  # V
    
    # sigma → KL: vertical up to top_bus, horizontal left (to kl), vertical down
    connect_points(d, cx(sigma), sigma['y'], cx(sigma), top_bus_y, 'arrow_skip')  # V
    connect_points(d, cx(sigma), top_bus_y, cx(kl), top_bus_y, 'arrow_skip')  # H
    connect_points(d, cx(kl), top_bus_y, cx(kl), kl['y'] + kl['h'], 'arrow_skip')  # V (reuse)
    
    # inp → Recon: vertical up to top_bus, horizontal right, vertical down
    connect_points(d, cx(inp), inp['y'], cx(inp), top_bus_y, 'arrow_skip')  # V
    connect_points(d, cx(inp), top_bus_y, cx(recon), top_bus_y, 'arrow_skip')  # H
    connect_points(d, cx(recon), top_bus_y, cx(recon), recon['y'] + recon['h'], 'arrow_skip')  # V
    
    # out → Recon: vertical up to top_bus, horizontal right (to recon), vertical down
    connect_points(d, cx(out), out['y'], cx(out), top_bus_y, 'arrow_skip')  # V
    connect_points(d, cx(out), top_bus_y, cx(recon), top_bus_y, 'arrow_skip')  # H
    connect_points(d, cx(recon), top_bus_y, cx(recon), recon['y'] + recon['h'], 'arrow_skip')  # V (reuse)
    return make_entry('VAE', d)


def model_gan():
    d = Diagram()
    noise = d.add_vertex('Noise z', 20, 92, 70, 26, 'noise')
    d.add_vertex('Generator G', 120, 52, 130, 185, 'container_purple')
    g1 = d.add_vertex('FC', 140, 87, 90, 28, 'fc')
    g2 = d.add_vertex('Deconv', 140, 139, 90, 28, 'deconv')
    fake = d.add_vertex('Fake x', 140, 191, 90, 26, 'output')
    real = d.add_vertex('Real x', 20, 262, 70, 28, 'input')
    d.add_vertex('Discriminator D', 340, 97, 144, 191, 'container_red')
    d1 = d.add_vertex('Conv', 360, 132, 104, 30, 'conv')
    d2 = d.add_vertex('FC', 360, 186, 104, 30, 'fc')
    dout = d.add_vertex('Real/Fake', 360, 240, 104, 28, 'sigmoid')

    connect_right(d, noise, g1)
    connect_down(d, g1, g2)
    connect_down(d, g2, fake)
    connect_points(d, right(real)[0], right(real)[1], left(d1)[0], left(d1)[1])
    # fake→d1 routing: explicit 4-segment path ABOVE real box (avoid crossing)
    # real box top at y=262, route at y=247 (15pt above)
    # Pattern: right → down → across → down to discriminator
    via_x = 245  # 15pt right of fake right edge (230)
    via_y = 247  # 15pt above real box top (262)
    connect_points(d, right(fake)[0], right(fake)[1], via_x, right(fake)[1], 'arrow_seg')      # Segment 1: right
    connect_points(d, via_x, right(fake)[1], via_x, via_y, 'arrow_seg')                        # Segment 2: down
    connect_points(d, via_x, via_y, left(d1)[0], via_y, 'arrow_seg')                           # Segment 3: across
    connect_points(d, left(d1)[0], via_y, left(d1)[0], left(d1)[1], 'arrow')                  # Segment 4: down with arrow
    connect_down(d, d1, d2)
    connect_down(d, d2, dout)
    return make_entry('GAN', d)


def model_dcgan_generator():
    d = Diagram()
    d.add_vertex('DCGAN Generator', 20, 13, 240, 413, 'container_purple')
    z = d.add_vertex('Noise z', 84, 48, 112, 26, 'noise')
    fc = d.add_vertex('FC + Reshape', 74, 98, 132, 30, 'fc')
    d1 = d.add_vertex('Deconv + BN + ReLU', 60, 152, 160, 32, 'deconv')
    d2 = d.add_vertex('Deconv + BN + ReLU', 60, 208, 160, 32, 'deconv')
    d3 = d.add_vertex('Deconv + BN + ReLU', 60, 264, 160, 32, 'deconv')
    d4 = d.add_vertex('Deconv + BN + ReLU', 60, 320, 160, 32, 'deconv')
    out = d.add_vertex('Output Image', 78, 376, 124, 30, 'output')
    for src, dst in [(z, fc), (fc, d1), (d1, d2), (d2, d3), (d3, d4), (d4, out)]:
        connect_down(d, src, dst)
    return make_entry('DCGAN Generator', d)


def model_dcgan_discriminator():
    d = Diagram()
    d.add_vertex('DCGAN Discriminator', 20, 13, 240, 411, 'container_red')
    inp = d.add_vertex('Input Image', 78, 48, 124, 30, 'input')
    c1 = d.add_vertex('Conv + LeakyReLU', 66, 102, 148, 32, 'conv')
    c2 = d.add_vertex('Conv + LeakyReLU', 66, 158, 148, 32, 'conv')
    c3 = d.add_vertex('Conv + LeakyReLU', 66, 214, 148, 32, 'conv')
    c4 = d.add_vertex('Conv + LeakyReLU', 66, 270, 148, 32, 'conv')
    flatten = d.add_vertex('Flatten', 88, 326, 104, 26, 'flatten')
    fc = d.add_vertex('FC', 102, 376, 76, 28, 'fc')
    out = d.add_vertex('Sigmoid', 94, 428, 92, 28, 'sigmoid')
    for src, dst in [(inp, c1), (c1, c2), (c2, c3), (c3, c4), (c4, flatten), (flatten, fc), (fc, out)]:
        connect_down(d, src, dst)
    return make_entry('DCGAN Discriminator', d)


def model_conditional_gan():
    d = Diagram()
    noise = d.add_vertex('z', 10, 60, 55, 24, 'noise')
    cond = d.add_vertex('c', 10, 124, 55, 24, 'seq_input')
    concat_g = d.add_vertex('C', 95, 92, 24, 24, 'concat')
    d.add_vertex('Generator G', 130, 51, 120, 133, 'container_purple')
    g = d.add_vertex('Gen', 150, 86, 80, 30, 'deconv')
    fake = d.add_vertex('Fake x', 150, 140, 80, 24, 'output')
    real = d.add_vertex('Real x', 10, 260, 70, 28, 'input')
    concat_fake = d.add_vertex('C', 280, 139, 26, 26, 'concat')
    concat_real = d.add_vertex('C', 280, 261, 26, 26, 'concat')
    d.add_vertex('Discriminator D', 335, 107, 130, 165, 'container_red')
    disc = d.add_vertex('Disc', 355, 142, 90, 30, 'conv')
    out = d.add_vertex('Real/Fake', 365, 224, 70, 28, 'sigmoid')

    connect_right(d, noise, concat_g)
    connect_right(d, cond, concat_g)
    connect_right(d, concat_g, g)
    connect_down(d, g, fake)
    connect_right(d, fake, concat_fake)
    connect_right(d, real, concat_real)
    cond_bus_x = 5
    cond_lx = cond['x']
    cond_ry = cy(cond)

    cf_lx, cf_ly = left(concat_fake)
    connect_points(d, cond_lx, cond_ry, cond_bus_x, cond_ry, 'arrow_seg')
    connect_points(d, cond_bus_x, cond_ry, cond_bus_x, cf_ly, 'arrow_seg')
    connect_points(d, cond_bus_x, cf_ly, cf_lx, cf_ly, 'arrow')

    cr_lx, cr_ly = left(concat_real)
    connect_points(d, cond_bus_x, cf_ly, cond_bus_x, cr_ly, 'arrow_seg')
    connect_points(d, cond_bus_x, cr_ly, cr_lx, cr_ly, 'arrow')
    disc_in_x, disc_in_y = left(disc)
    disc_bus_x = disc_in_x - 20
    for cnode in [concat_fake, concat_real]:
        sx, sy = right(cnode)
        connect_points(d, sx, sy, disc_bus_x, sy, 'arrow_seg')
        connect_points(d, disc_bus_x, sy, disc_bus_x, disc_in_y, 'arrow_seg')
    connect_points(d, disc_bus_x, disc_in_y, disc_in_x, disc_in_y)
    connect_down(d, disc, out)
    return make_entry('Conditional GAN', d)


def model_diffusion():
    d = Diagram()
    x0 = d.add_vertex('x0', 20, 35, 50, 28, 'input')
    x1 = d.add_vertex('x1', 100, 35, 50, 28, 'conv_sm')
    dots_top = d.add_vertex('...', 180, 35, 30, 28, 'ellipsis')
    xt = d.add_vertex('xt', 240, 35, 50, 28, 'noise')
    xt_rev = d.add_vertex('xt', 240, 190, 50, 28, 'noise')
    dots_bot = d.add_vertex('...', 180, 190, 30, 28, 'ellipsis')
    x1_rev = d.add_vertex('x1', 100, 190, 50, 28, 'conv_sm')
    x0_rev = d.add_vertex('x0', 20, 190, 50, 28, 'output')
    unet = d.add_vertex('U-Net Denoiser', 320, 112, 120, 36, 'deconv')

    connect_right(d, x0, x1)
    connect_right(d, x1, dots_top)
    connect_right(d, dots_top, xt)
    connect_points(d, right(xt_rev)[0], right(xt_rev)[1], left(dots_bot)[0], left(dots_bot)[1], 'arrow_label', 'denoise')
    connect_points(d, right(dots_bot)[0], right(dots_bot)[1], left(x1_rev)[0], left(x1_rev)[1])
    connect_points(d, right(x1_rev)[0], right(x1_rev)[1], left(x0_rev)[0], left(x0_rev)[1])
    connect_points(d, cx(xt), xt['y'] + xt['h'], cx(xt_rev), xt_rev['y'], 'arrow_label', 'forward noise')
    connect_right(d, xt_rev, unet)
    # Route from unet back to x1_rev using 4-segment H-V-H-V path (avoid crossing)
    unet_l_x, unet_l_y = left(unet)
    x1_rev_r_x, x1_rev_r_y = right(x1_rev)
    
    # Calculate via-points to route ABOVE the reverse flow
    via1_x = unet_l_x - 15  # 15pt left of unet
    via1_y = unet_l_y  # Same level as unet
    via2_y = x1_rev_r_y - 20  # 20pt above x1_rev (avoid crossing)
    
    # 4-segment routing: H (left from unet) → V (up) → H (left to align with x1_rev) → V (down to x1_rev)
    connect_points(d, unet_l_x, unet_l_y, via1_x, via1_y, 'arrow_seg')  # H: left from unet
    connect_points(d, via1_x, via1_y, via1_x, via2_y, 'arrow_seg')  # V: up
    connect_points(d, via1_x, via2_y, x1_rev_r_x + 30, via2_y, 'arrow_seg')  # H: across
    connect_points(d, x1_rev_r_x + 30, via2_y, x1_rev_r_x, x1_rev_r_y, 'arrow')  # V: down to x1_rev
    return make_entry('Diffusion Model', d)


def model_stacked_lstm():
    d = Diagram()
    inp = d.add_vertex('Sequence Input', 70, 20, 120, 32, 'seq_input')
    l1 = d.add_vertex('LSTM 1', 86, 76, 88, 34, 'lstm')
    l2 = d.add_vertex('LSTM 2', 86, 134, 88, 34, 'lstm')
    l3 = d.add_vertex('LSTM 3', 86, 192, 88, 34, 'lstm')
    fc = d.add_vertex('FC', 100, 250, 60, 30, 'fc')
    out = d.add_vertex('Output', 90, 304, 80, 30, 'output')
    for src, dst in [(inp, l1), (l1, l2), (l2, l3), (l3, fc), (fc, out)]:
        connect_down(d, src, dst)
    return make_entry('Stacked LSTM', d)


def model_bilstm():
    d = Diagram()
    inp = d.add_vertex('Sequence Input', 20, 104, 110, 32, 'seq_input')
    fwd = d.add_vertex('Forward LSTM', 160, 54, 110, 34, 'lstm')
    bwd = d.add_vertex('Backward LSTM', 160, 154, 110, 34, 'lstm')
    concat_node = d.add_vertex('C', 300, 104, 30, 30, 'concat')
    fc = d.add_vertex('FC', 360, 104, 60, 30, 'fc')
    out = d.add_vertex('Output', 450, 104, 70, 30, 'output')
    # Staggered routing: input fans to forward and backward LSTMs with 15pt offset
    inp_x, inp_y = right(inp)
    fwd_x, fwd_y = left(fwd)
    bwd_x, bwd_y = left(bwd)
    
    # Route 1: Direct connection to forward LSTM
    connect_points(d, inp_x, inp_y, fwd_x, fwd_y)
    
    # Route 2: Staggered connection to backward LSTM (offset by 15pt downward)
    via_y = inp_y + 15
    connect_points_via_y(d, inp_x, inp_y, bwd_x, bwd_y, via_y)
    connect_points(d, right(fwd)[0], right(fwd)[1], left(concat_node)[0], left(concat_node)[1])
    connect_points(d, right(bwd)[0], right(bwd)[1], left(concat_node)[0], left(concat_node)[1])
    connect_right(d, concat_node, fc)
    connect_right(d, fc, out)
    return make_entry('Bi-LSTM', d)


def model_stacked_gru():
    d = Diagram()
    inp = d.add_vertex('Sequence Input', 70, 20, 120, 32, 'seq_input')
    g1 = d.add_vertex('GRU 1', 92, 76, 76, 34, 'gru')
    g2 = d.add_vertex('GRU 2', 92, 134, 76, 34, 'gru')
    fc = d.add_vertex('FC', 100, 192, 60, 30, 'fc')
    out = d.add_vertex('Output', 90, 246, 80, 30, 'output')
    for src, dst in [(inp, g1), (g1, g2), (g2, fc), (fc, out)]:
        connect_down(d, src, dst)
    return make_entry('Stacked GRU', d)


def model_seq2seq():
    d = Diagram()
    inp = d.add_vertex('Input Seq', 10, 92, 70, 30, 'seq_input')
    d.add_vertex('Encoder LSTM', 90, 61, 184, 130, 'container_blue')
    e1 = d.add_vertex('h1', 110, 96, 28, 28, 'lstm')
    e2 = d.add_vertex('h2', 168, 96, 28, 28, 'lstm')
    e3 = d.add_vertex('hT', 226, 96, 28, 28, 'lstm')
    ctx = d.add_vertex('Context', 284, 92, 70, 30, 'fc')
    d.add_vertex('Decoder LSTM', 394, 61, 190, 130, 'container_green')
    d1 = d.add_vertex('y1', 414, 96, 30, 28, 'lstm')
    d2 = d.add_vertex('y2', 474, 96, 30, 28, 'lstm')
    d3 = d.add_vertex('yT', 534, 96, 30, 28, 'lstm')
    out = d.add_vertex('Output Seq', 594, 92, 70, 30, 'output')

    connect_right(d, inp, e1)
    connect_right(d, e1, e2)
    connect_right(d, e2, e3)
    connect_right(d, e3, ctx)
    connect_right(d, ctx, d1)
    connect_right(d, d1, d2)
    connect_right(d, d2, d3)
    connect_right(d, d3, out)
    return make_entry('Seq2Seq', d)


def model_seq2seq_attention():
    d = Diagram()
    inp = d.add_vertex('Input Seq', 10, 120, 70, 30, 'seq_input')
    d.add_vertex('Encoder', 100, 85, 184, 120, 'container_blue')
    e1 = d.add_vertex('h1', 120, 120, 28, 28, 'lstm')
    e2 = d.add_vertex('h2', 178, 120, 28, 28, 'lstm')
    e3 = d.add_vertex('h3', 236, 120, 28, 28, 'lstm')
    ctx = d.add_vertex('Context', 294, 120, 70, 30, 'fc')
    attn = d.add_vertex('Attention', 430, 30, 110, 30, 'self_attn')
    d.add_vertex('Decoder', 404, 85, 184, 140, 'container_green')
    d1 = d.add_vertex('y1', 424, 120, 28, 28, 'lstm')
    d2 = d.add_vertex('y2', 482, 120, 28, 28, 'lstm')
    d3 = d.add_vertex('y3', 540, 120, 28, 28, 'lstm')
    out = d.add_vertex('Output', 598, 120, 60, 30, 'output')

    connect_right(d, inp, e1)
    connect_right(d, e1, e2)
    connect_right(d, e2, e3)
    connect_right(d, e3, ctx)
    connect_right(d, ctx, d1)
    connect_right(d, d1, d2)
    connect_right(d, d2, d3)
    connect_right(d, d3, out)
    # Encoder→Attention fan-in: 3 encoders converge via vertical bus left of attention
    attn_top_y = attn['y']
    bus_x_enc = cx(attn) - 25  # Vertical bus left of attention
    for i, enc_cell in enumerate([e1, e2, e3]):
        enc_cx = cx(enc_cell)
        enc_t_y = enc_cell['y']
        via_x = bus_x_enc - (i * 15)  # Stagger: 0pt, -15pt, -30pt
        # Horizontal to bus
        connect_points(d, enc_cx, enc_t_y, via_x, enc_t_y, 'arrow_skip_seg')
        # Vertical to attention top
        connect_points(d, via_x, enc_t_y, via_x, attn_top_y, 'arrow_skip_seg')
        # Horizontal into attention
        connect_points(d, via_x, attn_top_y, cx(attn), attn_top_y, 'arrow_skip')

    # Attention→Decoder fan-out: attention bottom fans to 3 decoders via vertical bus right of attention
    attn_bot_y = attn['y'] + attn['h']
    bus_x_dec = cx(attn) + 25  # Vertical bus right of attention
    for i, dec_cell in enumerate([d1, d2, d3]):
        dec_cx = cx(dec_cell)
        dec_t_y = dec_cell['y']
        via_x = bus_x_dec + (i * 15)  # Stagger: 0pt, +15pt, +30pt
        # Horizontal to bus
        connect_points(d, cx(attn), attn_bot_y, via_x, attn_bot_y, 'arrow_skip_seg')
        # Vertical down to decoder level
        connect_points(d, via_x, attn_bot_y, via_x, dec_t_y, 'arrow_skip_seg')
        # Horizontal into decoder
        connect_points(d, via_x, dec_t_y, dec_cx, dec_t_y, 'arrow_skip')
    return make_entry('Seq2Seq + Attention', d)


def model_siamese_network():
    d = Diagram()
    in1 = d.add_vertex('Input A', 20, 80, 70, 30, 'input')
    in2 = d.add_vertex('Input B', 20, 170, 70, 30, 'input')
    d.add_vertex('Shared Encoder (=)', 140, 43, 110, 179, 'container_purple')
    enc1 = d.add_vertex('Encoder', 160, 78, 70, 34, 'encoder_h')
    enc2 = d.add_vertex('Encoder', 160, 168, 70, 34, 'encoder_h')
    emb1 = d.add_vertex('Emb A', 260, 80, 70, 30, 'embedding')
    emb2 = d.add_vertex('Emb B', 260, 170, 70, 30, 'embedding')
    dist = d.add_vertex('|a-b|', 360, 117, 46, 46, 'matmul')
    out = d.add_vertex('Similarity', 436, 123, 60, 34, 'output')

    connect_right(d, in1, enc1)
    connect_right(d, in2, enc2)
    connect_right(d, enc1, emb1)
    connect_right(d, enc2, emb2)
    # Staggered via-points for convergent edges (emb1→dist and emb2→dist)
    # Route 1 (emb1→dist): approaches via closer point
    emb1_x, emb1_y = right(emb1)
    dist_x, dist_y = left(dist)
    via1_x = dist_x - 20  # Approach closer
    connect_points(d, emb1_x, emb1_y, via1_x, emb1_y, 'arrow_seg')
    connect_points(d, via1_x, emb1_y, dist_x, dist_y, 'arrow')
    
    # Route 2 (emb2→dist): staggered approach (15pt further left)
    emb2_x, emb2_y = right(emb2)
    via2_x = dist_x - 35  # 15pt further left than route 1
    connect_points(d, emb2_x, emb2_y, via2_x, emb2_y, 'arrow_seg')
    connect_points(d, via2_x, emb2_y, via1_x, dist_y, 'arrow_seg')  # Approach via staggered point
    connect_points(d, via1_x, dist_y, dist_x, dist_y, 'arrow')
    connect_right(d, dist, out)
    return make_entry('Siamese Network', d)


def model_mlp_mixer():
    d = Diagram()
    inp = d.add_vertex('Patch Embedding', 20, 90, 82, 30, 'patch')
    d.add_vertex('Repeat xN', 112, 55, 367, 140, 'container_repeat')
    token = d.add_vertex('LN+Token MLP', 132, 90, 90, 30, 'ffn')
    add1 = d.add_vertex('+', 252, 92, 26, 26, 'add')
    channel = d.add_vertex('LN+Channel MLP', 308, 90, 95, 30, 'ffn')
    add2 = d.add_vertex('+', 433, 92, 26, 26, 'add')
    out = d.add_vertex('Output', 520, 88, 60, 30, 'output')

    connect_right(d, inp, token)
    connect_right(d, token, add1)
    connect_skip_horizontal(d, inp, add1)
    connect_right(d, add1, channel)
    connect_right(d, channel, add2)
    connect_skip_horizontal(d, add1, add2)
    connect_right(d, add2, out)
    return make_entry('MLP-Mixer', d)


def model_swin_block():
    d = Diagram()
    inp = d.add_vertex('Input', 130, 8, 80, 30, 'seq_input')
    d.add_vertex('Swin Block', 88, 39, 164, 493, 'container_purple')
    wmsa = d.add_vertex('W-MSA', 108, 74, 124, 32, 'mha')
    add1 = d.add_vertex('+', 155, 126, 30, 30, 'add')
    ln1 = d.add_vertex('LN', 130, 178, 80, 26, 'ln')
    swmsa = d.add_vertex('SW-MSA', 108, 228, 124, 32, 'mha')
    add2 = d.add_vertex('+', 155, 280, 30, 30, 'add')
    ln2 = d.add_vertex('LN', 130, 332, 80, 26, 'ln')
    mlp = d.add_vertex('MLP', 108, 382, 124, 32, 'ffn')
    add3 = d.add_vertex('+', 155, 434, 30, 30, 'add')
    ln3 = d.add_vertex('LN', 130, 486, 80, 26, 'ln')
    out = d.add_vertex('Output', 130, 538, 80, 30, 'output')

    for src, dst in [(inp, wmsa), (wmsa, add1), (add1, ln1), (ln1, swmsa), (swmsa, add2), (add2, ln2), (ln2, mlp), (mlp, add3), (add3, ln3), (ln3, out)]:
        connect_down(d, src, dst)
    connect_skip_vertical(d, inp, add1)
    connect_skip_vertical(d, ln1, add2)
    connect_skip_vertical(d, ln2, add3)
    return make_entry('Swin Transformer Block', d)


def model_efficientnet_mbconv():
    d = Diagram()
    inp = d.add_vertex('Input', 170, 10, 80, 30, 'seq_input')
    d.add_vertex('MBConv', 28, 40, 306, 355, 'container_green')
    expand = d.add_vertex('Conv 1x1 Expand', 142, 75, 136, 30, 'conv_sm')
    dw = d.add_vertex('DW Conv 3x3', 150, 129, 120, 30, 'conv_sm')
    gap = d.add_vertex('GAP', 48, 183, 72, 26, 'gap')
    fc = d.add_vertex('FC', 58, 231, 52, 24, 'fc')
    sig = d.add_vertex('Sigmoid', 46, 277, 76, 24, 'sigmoid')
    mul = d.add_vertex('*', 260, 183, 30, 30, 'multiply')
    proj = d.add_vertex('Conv 1x1 Project', 138, 237, 144, 30, 'conv_sm')
    add = d.add_vertex('+', 260, 291, 28, 28, 'add')
    out = d.add_vertex('Output', 234, 345, 80, 30, 'output')

    connect_down(d, inp, expand)
    connect_down(d, expand, dw)
    connect_down(d, dw, mul)
    connect_down(d, mul, proj)
    connect_down(d, proj, add)
    connect_down(d, add, out)
    
    # SE block routes via FAR LEFT (x=5) to avoid crossing skip connection
    # GAP: from right of dw → far left bus → down
    dw_rx, dw_ry = right(dw)
    gap_lx, gap_ly = left(gap)
    connect_points_via_x(d, dw_rx, dw_ry, gap_lx, gap_ly, 5, 'arrow')
    
    connect_down(d, gap, fc)
    connect_down(d, fc, sig)
    
    # Sigmoid to multiply: routes far left then right
    sig_lx, sig_ly = left(sig)
    mul_lx, mul_ly = left(mul)
    connect_points_via_x(d, sig_lx, sig_ly, mul_lx, mul_ly, 5, 'arrow')
    
    # Skip connection routes via FAR RIGHT (x=334, right of MBConv container at x=28+306=334)
    # Input → far right bus → down → add
    inp_rx, inp_ry = right(inp)
    add_tx, add_ty = top(add)
    skip_bus_x = 334 + 30  # 30pt margin beyond container
    connect_points(d, inp_rx, inp_ry, skip_bus_x, inp_ry, 'arrow_seg')
    connect_points(d, skip_bus_x, inp_ry, skip_bus_x, add_ty - 10, 'arrow_seg')
    connect_points(d, skip_bus_x, add_ty - 10, add_tx, add_ty - 10, 'arrow_seg')
    connect_points(d, add_tx, add_ty - 10, add_tx, add_ty, 'arrow_skip')
    return make_entry('EfficientNet MBConv', d)


def model_gat_layer():
    d = Diagram()
    ni = d.add_vertex('Node i', 20, 90, 70, 30, 'seq_input')
    nj = d.add_vertex('Node j', 20, 40, 70, 30, 'seq_input')
    nk = d.add_vertex('Node k', 20, 140, 70, 30, 'seq_input')
    attn = d.add_vertex('Attn Weights', 130, 90, 90, 30, 'self_attn')
    mul = d.add_vertex('*', 250, 95, 30, 30, 'multiply')
    agg = d.add_vertex('+', 310, 95, 30, 30, 'add')
    act = d.add_vertex('Activation', 370, 92, 65, 30, 'gelu')
    out = d.add_vertex('Output Feat', 465, 90, 70, 32, 'output')

    connect_right(d, ni, attn)
    connect_right(d, attn, mul)
    # Staggered fan-in to multiply node (nj, nk)
    mul_left_x, mul_left_y = left(mul)
    via_x1 = mul_left_x - 20  # Route 1: nj - closer to mul
    via_x2 = mul_left_x - 35  # Route 2: nk - staggered by 15pt
    
    nj_x, nj_y = right(nj)
    nk_x, nk_y = right(nk)
    
    connect_points_via_x(d, nj_x, nj_y, mul_left_x, mul_left_y, via_x1)
    connect_points_via_x(d, nk_x, nk_y, mul_left_x, mul_left_y, via_x2)
    
    connect_right(d, mul, agg)
    
    # Staggered fan-in to add node (ni)
    agg_left_x, agg_left_y = left(agg)
    via_x3 = agg_left_x - 20
    ni_x, ni_y = right(ni)
    connect_points_via_x(d, ni_x, ni_y, agg_left_x, agg_left_y, via_x3)
    connect_right(d, agg, act)
    connect_right(d, act, out)
    return make_entry('GAT Layer', d)


def build_models():
    models = [
        model_lenet5(),
        model_alexnet(),
        model_vgg_block(),
        model_resnet_block(),
        model_resnet_bottleneck(),
        model_densenet_block(),
        model_inception_module(),
        model_mobilenet_block(),
        model_unet(),
        model_fpn(),
        model_yolo_head(),
        model_faster_rcnn(),
        model_fcn_decoder(),
        model_aspp_module(),
        model_transformer_encoder(),
        model_transformer_decoder(),
        model_full_transformer(),
        model_vit(),
        model_bert_block(),
        model_gpt_block(),
        model_multi_head_attention_detail(),
        model_autoencoder(),
        model_vae(),
        model_gan(),
        model_dcgan_generator(),
        model_dcgan_discriminator(),
        model_conditional_gan(),
        model_diffusion(),
        model_stacked_lstm(),
        model_bilstm(),
        model_stacked_gru(),
        model_seq2seq(),
        model_seq2seq_attention(),
        model_siamese_network(),
        model_mlp_mixer(),
        model_swin_block(),
        model_efficientnet_mbconv(),
        model_gat_layer(),
    ]
    if len(models) != 38:
        raise RuntimeError(f'Expected 38 models, got {len(models)}')
    return models


def main():
    output_path = '/Users/huan/Downloads/drawio-hub/ieee_trans_nn_model_library.xml'
    models = build_models()
    payload = build_library(models)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(payload)
    print(f'Wrote {output_path} with {len(models)} models')


if __name__ == '__main__':
    main()
