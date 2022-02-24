import os.path as osp
import matplotlib.pyplot as plt
# from bokeh.palettes import Category20
from sklearn.manifold import TSNE
import pandas as pd


def tsne(feature_map, results, component_num, dir_path):
    # fig, ax = plt.subplots()
    # y_pred, y, conf, img_name = results
    y_pred, y  = results
    model_tsne = TSNE(n_components=component_num, random_state=40)
    model_tsne.fit(feature_map)
    embeddings = model_tsne.embedding_

    df = pd.DataFrame()
    df['tsne0'] = embeddings[:, 0]
    df['tsne1'] = embeddings[:, 1]
    df['gt'] = y
    # df['net_label_prediction'] = results[]
    # df['net_corona_confidence'] = conf[:, 0]
    # df['net_no_corona_confidence'] = conf[:, 1]
    # df['image_name'] = img_name

    df_path = osp.join(dir_path, "tsne_results.csv")
    df.to_csv(df_path)

    # for label in [0, 1]:  #
    #     df_per_label = df[df['gt'] == label]
    #     plt.scatter(df_per_label['tsne0'], df_per_label['tsne1'], s=1)
    plot_tsne(features_df=df, dir_path=dir_path)


def plot_tsne(features_df, dir_path, hover=False):
    unique_gt = features_df['gt'].unique()
    unique_gtnames = ['positive for Covid-19' if g == 1 else 'negative for Covid-19' for g in unique_gt]
    # colors = Category20[20]

    features_df['gt_name'] = ['positive for Covid-19' if g == 1 else 'negative for Covid-19' for g in features_df['gt']]

    # if hover:
    #     features_df['images_b64'] = [base64.b64encode(open(img, 'rb').read()).decode('utf-8') for img in
    #                                  features_df['image_path']]
    #     tips = """
    #             <div>
    #                 <div>
    #                     <img
    #                         src="data:image/jpeg;base64,@images_b64" height=150 width=150
    #                         style="float: left; margin: 0px 2px 2px 0px"
    #                         border="2"
    #                     ></img>
    #                 </div>
    #
    #                 <div>
    #                     <span style="font-size: 15px;">@gt_name</span>
    #                 </div>
    #                 <div>
    #                     <span style="font-size: 15px;">@score</span>
    #                 </div>
    #             </div>
    #         """
    # else:
    #     tips = """
    #             <div>
    #                 <div>
    #                     <span style="font-size: 50px;">@gt_name</span>
    #                 </div>
    #                 <div>
    #                     <span style="font-size: 50px;">@net_corona_confidence</span>
    #                 </div>
    #                 <div>
    #                     <span style="font-size: 35px;">@image_name</span>
    #                 </div>
    #             </div>
    #         """
    #
    # fig_to_plot = figure(width=2000, height=1500, tooltips=tips)
    #
    # # Usable if we want to give a sense of confidence
    # # features_df['error'] = abs(features_df['gt'] - features_df['net_no_corona_confidence'])
    # # features_df['uncertain'] = 2 * (0.5 - abs(0.5 - features_df['net_no_corona_confidence']))
    # # features_df['line_alpha_error'] = 0.1 + 0.9 * features_df['error']
    # # features_df['line_alpha_uncertain'] = 0.1 + 0.9 * features_df['uncertain']
    #
    # for label, label_name in zip(unique_gt, unique_gtnames):
    #     df_per_label = features_df[features_df['gt'] == label]
    #     source = ColumnDataSource(df_per_label)
    #
    #     # Usable if we want to give a sense of confidence
    #     # fig_to_plot.scatter('tsne0', 'tsne1', color=colors[label * 19], source=source,
    #     #                     legend_label='legend_label: ' + str(label),
    #     #                     fill_alpha='uncertain', line_alpha='line_alpha_uncertain', size=30)
    #
    #     fig_to_plot.scatter('tsne0', 'tsne1', color=colors[label * 19], source=source,
    #                         legend_label=label_name, size=30)
    #
    # fig_to_plot.legend.location = 'bottom_left'
    # fig_to_plot.legend.click_policy = 'hide'
    # fig_to_plot.legend.label_text_font_size = '50pt'
    #
    # html_path = osp.join(dir_path, '{}.html'.format('tnse_plot'))
    # if osp.exists(html_path):
    #     os.remove(html_path)
    #
    # output_file(html_path, mode='inline')
    # save(fig_to_plot)

    # save tsne as png
    for label, label_name in zip(unique_gt, unique_gtnames):
        df_per_label = features_df[features_df['gt'] == label]
        plt.scatter(df_per_label['tsne0'], df_per_label['tsne1'],
                    #c=colors[label * 19],
                    label=label_name, alpha=0.4, s=30)
        plt.legend(scatterpoints=1, loc="lower left")
        plt.title('T-SNE plot')
        plt.xlabel('tsne0')
        plt.ylabel('tsne1')
    plt.savefig(fname=osp.join(dir_path, 'tsne_plot.png'), format="png", bbox_inches="tight")


