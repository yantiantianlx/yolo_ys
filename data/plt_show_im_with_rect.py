
import matplotlib.pyplot as plt


def plt_show(im, rect_list, name_list=None, conf_list=None):
    if name_list is not None:
        assert len(rect_list) == len(name_list)
    if conf_list is not None:
        assert len(rect_list) == len(conf_list)

    fig = plt.figure(frameon=False)
    dpi = 100
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    for i in range(len(rect_list)):
        bbox = rect_list[i]
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='w', linewidth=3, alpha=0.5))

        name = '' if name_list is None else str(name_list[i])
        conf = '' if conf_list is None else str(conf_list[i])
        bbox_str = name + '  ' + conf

        ax.text(bbox[0], bbox[1] - 2, bbox_str, fontsize=5, family='serif', bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'), color='white')

    plt.show()


if __name__ == '__main__':
    import cv2

    im_path = '111.jpg'
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    plt_show(im, [[20, 20, 50, 50]], [1])

