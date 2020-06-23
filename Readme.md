# 愛文芒果等級分類競賽
![](https://i.imgur.com/4PoXrBb.jpg)
* [比賽官網](https://aidea-web.tw/topic/72f6ea6a-9300-445a-bedc-9e9f27d91b1c)
* [工作坊競賽教學影片](https://www.youtube.com/playlist?list=PLJ6QzDAugy1muFIHX17go-OR62avvWr1A)
* 活動時間
    
    | 時間                                                       |            等級分類競賽事件            |
    | ---------------------------------------------------------- |:--------------------------------------:|
    | 2020/03/02                                                 |   公布初賽訓練集、建構集資料開放下載   |
    | 2020/03/09                                                 |   公布建構集Baseline與演算法等 參數    |
    | 2020/02/03-05/15                                           |   開放報名（提供註冊及Sample data）    |
    | 2020/06/12                                                 | 公布初賽測試集，開放下載及上傳答案算分 |
    | 2020/06/16 23:59:59	|初賽截止，關閉測試集的資料上傳答案功能                                         
    | 2020/06/24	|公布初賽成績                                    |                                        
    | 2020/07/10	|公布決賽訓練集、建構集資料開放下載              |                                        
    |2020/07/17	|公布建構集Baseline與演算法等參數|
    |2020/10/30	|併隊截止|
    |2020/11/10	|公布決賽測試集，開放下載及上傳答案算分|
    |2020/11/15 |23:59:59	決賽截止，關閉測試集的資料上傳答案功能|
    |2020/12/02|	公布決賽成績(系統分數)，開始上傳報告|
    |2020/12/15| 23:59:59	上傳報告截止，開始評估（系統+報告）|
    |2020/12/29 |	公布最終成績（系統+報告）|
    
* [訓練資料集](https://drive.google.com/open?id=1Kqblc0Z4PKYzxXIF2jARgyeft22QQcWv)
* [共用雲端](https://drive.google.com/drive/u/1/folders/0AHiJevojRo9vUk9PVA)
* [程式碼專區](https://github.com/angelowen/mango-killer)

---
* 操作方法:
    * 下載訓練資料置於目錄下
    * python train_vgg16.py -> 普通版
    * python test.py -> 記得更改 PATH_TO_WEIGHTS = './model-0.XX-best_train_acc.pth'
    * write_answer.py -> 答案寫入csv檔

## 實作技巧
* Cyclical Learning Rates and momentum
* data augmentation
* mixup
* RandomAugment
* AutoAugment
## 嘗試模型:
* vgg16(pretrained,no-pretrained)
* vgg19(pretrained,no-pretrained)
* SENet
* googlenet
* efficientnet(b7、b8)
* SVM
## 結論
* efficientnet b7 效果最好
* googlenet 也不錯
* svm可用1*1 conv解決RGB三維問題
* SEnet存model時有警告

## Reference:
### 參數設定
https://www.jishuwen.com/d/2B5Z/zh-tw

### Data Augumentation
https://zhuanlan.zhihu.com/p/53367135

https://www.itread01.com/content/1545405497.html

https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-transform/

https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomAffine

https://blog.csdn.net/qq_32768091/article/details/78735140
### Cyclical Learning Rates,momentum
https://blog.csdn.net/weixin_43896398/article/details/84762886

https://shenxiaohai.me/2019/02/27/lr_find/
###  RandomAugment
https://github.com/ildoonet/pytorch-randaugment
### AutoAugment
https://github.com/tensorflow/models/tree/master/research/autoaugment

https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/train.py

### Cutout Mixup Labelsmoothing 實作方法
https://blog.csdn.net/winycg/article/details/88410981

### gridmask實作
https://www.kaggle.com/haqishen/gridmask

### augmix 實作
https://github.com/google-research/augmix/blob/master/cifar.py

### Cutmix
https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py

### Mixup
https://github.com/facebookresearch/mixup-cifar10

### augmentor
https://github.com/mdbloice/Augmentor

### 其他數據增強方法
http://mirlab.org/users/yihsuan.chen/paper/untitled.pdf

https://zhuanlan.zhihu.com/p/104992391

https://github.com/CrazyVertigo/awesome-data-augmentation

Fmix可嘗試
https://github.com/ecs-vlc/FMix/

