


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# csv_file = "./decision_result_test.xlsx"
# csv_data = pd.read_excel(csv_file,'Sheet1',8)
# df = pd.DataFrame(csv_data)
# print(df)

# df.plot(x = 'window_length', y = 'mean', color = 'blue', marker = 'o',  label='mean')
# df.plot(x = 'window_length', y = 'raw', color = 'red', marker = 'o', label='raw')
# df.plot(x = 'window_length', y = 'hybrid', color = 'green', marker = 'o', legend = False)
# df.plot(x = 'window_length', y = 'histogram', color = 'purple', marker = 'o', legend = False)
# df.plot(x = 'window_length', y = 'no_arbitration', color = 'grey', marker = 'o', legend = False)

#71ae46 - 96b744 - c4cc38 - ebe12a - eab026 - e3852b - d85d2a - ce2626 - ac2026 - 71ae46 - 96b744 - c4cc38
def plotline(csv_file, name,y_ticks):
        csv_data1 = pd.read_excel(csv_file, name)
        df = pd.DataFrame(csv_data1)
        fig, ax = plt.subplots(figsize = (16, 9))

        plt.plot(df.loc[:,'window_length'], df.loc[:,'no_arbitration',].mul(100), c='#71ae46', marker='s', linestyle='--', label='no_arbitration',linewidth =4.0,markersize=15)
        plt.plot(df.loc[:,'window_length'], df.loc[:,'mean',].mul(100), c='#c4cc38', marker='o', linestyle='--', label='mean',linewidth =4.0,markersize=15)
        plt.plot(df.loc[:,'window_length'], df.loc[:,'raw',].mul(100), c='#eab026', marker='^', linestyle='-', label='raw',linewidth =4.0,markersize=15)
        plt.plot(df.loc[:,'window_length'], df.loc[:,'histogram',].mul(100), c='#d85d2a', marker='v', linestyle='-', label='histogram',linewidth =4.0,markersize=15)
        plt.plot(df.loc[:,'window_length'], df.loc[:,'hybrid',].mul(100), c='#ac2026', marker='D', linestyle='-', label='hybrid',linewidth =4.0,markersize=15)
        # plt.axis('off')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # xmin, xmax = ax.get_xlim()
        # ymin, ymax = ax.get_ylim()
        # x_adjustment = (xmax - xmin) * 0.1
        # y_adjustment = (ymax - ymin) * 0.1
        # ax.axis([xmin, xmax, ymin-y_adjustment, ymax+y_adjustment])
        plt.grid(axis='y')
        plt.legend(fontsize=20)
        x_ticks=[60,180,300,400,600]
        plt.xticks(ticks=x_ticks,fontsize=30)
        ax.tick_params(pad=30)
        # y_ticks=[84,86,88,90,92,94]
        # y_ticks=[68,76,84,92,100]

        plt.yticks(ticks=y_ticks,fontsize=30)

        plt.xlabel('Window length (second)', fontsize=40, color='k',labelpad=30)

        plt.ylabel('Specificity (%)', fontsize=40, color='k',labelpad=30)
        plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
        plt.show()


# csv_data1 = pd.read_excel(csv_file,'Sheet2')
# df1 = pd.DataFrame(csv_data1)
# print(df1)
def plotbox(csv_file,name,y_ticks):
        csv_data1 = pd.read_excel(csv_file, name)
        df1 = pd.DataFrame(csv_data1)
        print(df1)
        fig, ax = plt.subplots(figsize = (16, 9))
        box = ax.boxplot(df1.mul(100),patch_artist=True,capprops={'linewidth':2},whiskerprops={'linewidth':2},medianprops={'color':'#e3852b','linewidth':3},showmeans=True,meanline=True,meanprops={'color':'#e3852b','linewidth':3, 'linestyle':'--'})

        color = ['#71ae46' ,'#c4cc38','#eab026' ,'#d85d2a','#ac2026']
        color1=['#00994e','#96b744', '#ebe12a', '#e3852b', '#ce2626',]

        for box1, c in zip(box['medians'], color1):

                box1.set(color=c)
        for box3, c in zip(box['means'], color1):

                box3.set(color=c)



        for box2, c in zip(box['boxes'], color):
                # 箱体边框颜色
                box2.set(color='black', linewidth=2)
                # 箱体内部填充颜色
                box2.set(facecolor=c)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.grid(axis='y')
        # plt.legend(fontsize=20)
        x_ticks=['no arbitration','mean','raw','histogram','hybrid']
        plt.xticks(ticks=[1,2,3,4,5],labels=x_ticks,fontsize=30)
        ax.tick_params(pad=30)
        # y_ticks=[87,89,91,93,95,97]
        # y_ticks=[81,83,85,87,89,91]

        plt.yticks(ticks=y_ticks,fontsize=30)

        plt.xlabel('Arbitration Method', fontsize=40, color='k',labelpad=30)

        plt.ylabel('Accuracy (%)', fontsize=40, color='k', labelpad=30)
        plt.savefig(name+'.png', dpi=300, bbox_inches='tight')

        plt.show()



def plotscatter(csv_file,name,y_ticks):
        csv_data1 = pd.read_excel(csv_file, name)
        df1 = pd.DataFrame(csv_data1)
        print(df1)
        # print(df1.columns)
        fig, ax = plt.subplots(figsize=(16, 9))
        markers=['s','o','^','v','D']
        color=['#3b6291','#943c39','#779043','#624c7c','#388498','#bf7334']
        # color = ['#71ae46' ,'#c4cc38','#eab026' ,'#d85d2a','#ac2026']

        for i in df1.index:
                x=[1, 2, 3, 4, 5]
                x=[item + (i%5)*0.05 for item in x]
                plt.scatter(x,df1.loc[i].mul(100),marker=markers[i//5],color='white',s=500,edgecolors=color[i//5],linewidths=5)
        # plt.scatter([1.1,2.1,3.1,4.1,5.1],  df1.mean(axis=0).mul(100), color='#bf7334', marker='_')
        x=[1.1,2.1,3.1,4.1,5.1]
        for idx,i in enumerate(df1.mean(axis=0).mul(100)):
                plt.hlines(y=i, xmin=x[idx]-0.2, xmax=x[idx]+0.2, color='black',linestyle='--',linewidth =4.0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.grid(axis='y')
        # plt.legend(fontsize=20)
        x_ticks=['no arbitration','mean','raw','histogram','hybrid']
        plt.xticks(ticks=[1.1,2.1,3.1,4.1,5.1],labels=x_ticks,fontsize=30)
        ax.tick_params(pad=30)
        # y_ticks=[87,89,91,93,95,97]
        # y_ticks=[81,83,85,87,89,91]

        plt.yticks(ticks=y_ticks,fontsize=30)

        plt.xlabel('Arbitration Method', fontsize=40, color='k',labelpad=30)

        plt.ylabel('Accuracy (%)', fontsize=40, color='k', labelpad=30)
        plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
        plt.show()

csv_file = "./decision_result_test.xlsx"

plotline(csv_file,'specificity',[88,90,92,94,96,98,100])
# plotline(csv_file,'sensitivity',[68,76,84,92,100])
# plotbox(csv_file,'deep4_tuab_60s',[81,83,85,87,89,91])
# plotbox(csv_file,'deep4_tuab_600s',[87,89,91,93,95,97])

# plotscatter(csv_file,'tcn_tuab_60s',[83,85,87,89,91],shift=True)
# plotscatter(csv_file,'vit_small_tuab_60s',[71,73,75,77,79,81,83])
# plotscatter(csv_file,'vit_tiny_tuab_60s',[83,85,87,89,91,93])


# def ave(table):
#     grouped = table.groupby(by=df["start"])
#     averaged=grouped.mean()
#     return averaged
# # plt.show()
#
# hybrid=df.loc[(df["use_hybrid"]==True)&(df['use_his']==True),['ave_test_acc','start']]
# print(hybrid)
#
# hybrid_ave=ave(hybrid)
# print('hybrid_ave',hybrid_ave)
#
# histogram=df.loc[(df["use_hybrid"]==False)&(df['use_his']==True),['ave_test_acc']]
# raw=df.loc[(df["use_hybrid"]==False)&(df['use_his']==False),['ave_test_acc']]
# mean=df.loc[(df["use_hybrid"]==False)&(df['use_his']==False),['ave_mean_acc']]
# no_arb=df.loc[(df["use_hybrid"]==False)&(df['use_his']==False),['ave_ori_acc']]
#
# histogram_ave=ave(histogram)
# sort_list=[2008,1530]
# print('histogram_ave',histogram_ave.loc[sort_list])
#
# raw_ave=ave(raw)
# print('raw_ave',raw_ave)
#
# mean_ave=ave(mean)
# print('mean_ave',mean_ave)
#
# no_ar_ave=ave(no_arb)
# print('no_ar_ave',no_ar_ave)
#
#
# from matplotlib.pyplot import MultipleLocator
#
# lambda1 = [0.05, 0.1, 0.2, 0.5, 0.6]
# accuracy = [93.99, 93.34, 93.09, 92.97, 91.77]
# flops = [56.63, 62.27, 75.76, 78.78, 85.82]
# params = [58.96, 61.27, 73.99, 76.88, 84.97]
# plt.plot(lambda1, flops, c='blue', marker='o', linestyle=':', label='FLOPs')
# plt.plot(lambda1, accuracy, c='red', marker='*', linestyle='-', label='Accuracy')
# plt.plot(lambda1, params, c='green', marker='+', linestyle='--', label='parameters')
#
# #设置图例并且设置图例的字体及大小
# font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
# plt.xticks(fontproperties = 'Times New Roman',fontsize=10)
# plt.yticks(fontproperties = 'Times New Roman',fontsize=10)
#
# plt.xlabel(u'λ', font1)
# plt.ylabel(u'Pruned Percentage & Accuracy (%)', font1)
#
# # 图例展示位置，数字代表第几象限
# plt.legend(loc=4, prop=font1)
#
# # Axes(ax)对象，主要操作两个坐标轴间距
# x_major_locator = MultipleLocator(0.05)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# plt.show()
