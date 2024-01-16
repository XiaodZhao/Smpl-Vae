from xml.parsers.expat import model
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# plt.switch_backend('agg')
import numpy as np

def display_model(
        model_info,
        model_faces=None,
        with_joints=False,
        kintree_table=None,
        ax=None,
        show=True,
        savepath=None,
        only_joint=False,
        pNum = 1,
        Origin = None,
        d_colors = None):
    """
    Displays mesh batch_idx in batch of model_info, model_info as returned by
    generate_random_model
    """
    if ax is None:
        fig_size = [9, 6.8]
        fig = plt.figure(figsize=fig_size)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
        ax = fig.add_subplot(111, projection='3d')
    verts_ = model_info['verts']
    joints_ = model_info['joints']
    labels_ = []
    pol = []
    for n in range(pNum):

        origin =Origin[n]
        verts = verts_[n] * np.tile(np.array([1., 1., 1.]), (verts_[n].shape[0], 1))
        verts[:, [0, 2]] = verts[:, [2, 0]]
        verts[:, [1, 2]] = verts[:, [2, 1]]
        verts = verts + np.tile(origin, (verts.shape[0], 1))

        joints = joints_[n] * np.tile(np.array([1., 1., 1.]), (joints_[n].shape[0], 1))
        joints[:, [0, 2]] = joints[:, [2, 0]]
        joints[:, [1, 2]] = joints[:, [2, 1]]
        joints = joints + np.tile(origin, (joints.shape[0], 1))

        if model_faces is None:
            print('model facesssssssssssssssssssssssss')
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],  alpha=0.3)
        elif not only_joint:
            print('joints')
            mesh = Poly3DCollection(verts[model_faces], alpha=0.3,label =f'id_{n}')
            face_color = d_colors[n]
            edge_color = (50 / 255, 50 / 255, 50 / 255)
            mesh.set_edgecolor(edge_color)
            mesh.set_facecolor(face_color)
            pol.append(mesh)
            ax.add_collection3d(mesh)
            labels_.append( f'id_{n}')
            # ax.legend()
        colors = []
        left_right_mid = ['r', 'g', 'b']
        kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
        for c in kintree_colors:
                colors += left_right_mid[c]
        # For each 24 joint
        for i in range(1, kintree_table.shape[1]):
                j1 = kintree_table[0][i]
                j2 = kintree_table[1][i]
                ax.plot([joints[j1, 0], joints[j2, 0]],
                        [joints[j1, 1], joints[j2, 1]],
                        [joints[j1, 2], joints[j2, 2]],
                        color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
        # if with_joints :
        #     ax = draw_skeleton(joints, kintree_table=kintree_table, ax=ax)
    # ax.legend(pol,labels_,loc='best')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0.2, 2)
#     ax.legend(pol,labels_)
    # ax.view_init(azim=-90, elev=100)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if savepath:
        # print('Saving figure at {}.'.format(savepath))
        plt.savefig(savepath, dpi=300)
    if show:
        plt.show()
    plt.close()
    return ax


def draw_skeleton(joints3D, kintree_table, ax=None, with_numbers=False):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    colors = []
    left_right_mid = ['r', 'g', 'b']
    kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
    for c in kintree_colors:
        colors += left_right_mid[c]
    # For each 24 joint
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                [joints3D[j1, 1], joints3D[j2, 1]],
                [joints3D[j1, 2], joints3D[j2, 2]],
                color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_numbers:
            ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    return ax
